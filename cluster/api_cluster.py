import httpx
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, joinedload
import yaml
from orm import OutputEquivalenceCluster, Tensorflow, Pytorch

with open('config.yml', 'r', encoding='utf-8') as file:  # 读取config.yml文件
    config = yaml.safe_load(file)
# 从配置中提取数据库连接信息
db_config = config['db']['mysql']
host = db_config['host']
user = db_config['user']
password = db_config['password']
database = db_config['database']
db_url = f"mysql+pymysql://{user}:{password}@{host}/{database}"  # 创建数据库连接字符串
# 创建数据库连接
engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
session = Session()

# 设置代理
proxy = httpx.Client(proxies={
    "http://": "http://127.0.0.1:7890",
    "https://": "http://127.0.0.1:7890"
})
openai_client = OpenAI(api_key=config['openai']['api_key'], http_client=proxy)


def output_equivalence_cluster(openai_client, pytorch_api):
    # 获取所有TensorFlow API 名称并添加入Prompt
    tensorflow_apis = session.query(Tensorflow).all()
    tensorflow_api_names = [api.name for api in tensorflow_apis]
    prompt = f"TensorFlow API List:\n{tensorflow_api_names}\nFor Pytorch's {pytorch_api.name}, which APIs or combinations of APIs from the above TensorFlow have the same return value? Please output the function names in JSON format following these examples:\nExample 1:\n{{\n    1 : [\"tf.keras.losses.CategoricalCrossentropy\",\"tf.one_hot\"],\n    2 : [\"tf.nn.softmax_cross_entropy_with_logits\",\"tf.one_hot\"]\n}}\nExample 2:\n{{\n    # When there are no TensorFlow APIs whose combined output with Pytorch's API has the same value, output an empty json\n}}"
    try:
        # 调用 OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        # 解析返回的 JSON 数据
        output_apis = response.choices[0].message.content
        print(f"Output APIs: {output_apis}")
        if output_apis:
            # 解析 JSON 并创建数据库对象
            result_dict = eval(output_apis)
            new_cluster = OutputEquivalenceCluster()
            session.add(new_cluster)

            # 遍历结果字典，并将结果添加到数据库
            for key, api_list in result_dict.items():
                for api_name in api_list:
                    # 查询 TensorFlow API
                    tf_api = session.query(Tensorflow).filter(Tensorflow.name == api_name).first()
                    if tf_api:
                        # 建立 TensorFlow 和 OutputEquivalenceCluster 的关联
                        new_cluster.tensorflows.append(tf_api)
            session.commit()  # 提交事务
            print("Output equivalence cluster created successfully.")
        else:
            print("No equivalent TensorFlow APIs found.")

    except Exception as e:
        session.rollback()  # 回滚在异常中的任何数据库更改
        print(f"An error occurred: {e}")


pytorch_apis = session.query(Pytorch).options(joinedload(Pytorch.output_clusters)).all()
unclustered_apis = [api for api in pytorch_apis if not api.output_clusters]
count = 0
for pytorch_api in unclustered_apis[:50]: # 对前50个未聚类的 Pytorch API 进行聚类
    count += 1
    output_equivalence_cluster(openai_client, pytorch_api)
    print(f"Processing API {count}/{len(unclustered_apis)}: {pytorch_api.name}")

