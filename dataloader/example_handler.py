from utils import get_session, get_llm_client
from orm import API

session = get_session()
llm = get_llm_client(llm='gpt4o-mini')

# 首先从数据库中获取所有is_example_handled为False的API
apis = session.query(API).filter_by(is_example_handled=False).all()
for api in apis:
    try:
        if not api.example:
            api.is_example_handled = True
            session.commit()
            continue
        #TODO 处理API的示例代码


    except Exception as e:
        session.rollback()
        print(f"Error occurred while handling examples for APIs: {e}")
