import importlib
import inspect
import ast
import os
from json import JSONDecodeError


class APIChecker:
    def __init__(self):
        self.errors = []

    def handle_module_alias(self, module_name):
        return module_name

    def extract_api_calls(self, file_content):
        tree = ast.parse(file_content)
        api_calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                module_name = node.func.value.id
                api_name = node.func.attr
                full_api_name = f"{module_name}.{api_name}"
                api_calls.append(full_api_name)
        
        return api_calls

    def validate_api(self, full_api_name):
        module_name = ""
        api_name = ""
        try:
            module_name, api_name = full_api_name.rsplit('.', 1)
            module_name = self.handle_module_alias(module_name)
            module = importlib.import_module(module_name)
            func = getattr(module, api_name, None)
            if inspect.ismodule(func):
                self.errors.append(f"{full_api_name} is a module, not a function.")
                return False
            if inspect.isclass(func):
                self.errors.append(f"{full_api_name} is a class, not a function.")
                return False
            return True
        except ImportError as e:
            self.errors.append(f"Module {module_name} not found: {str(e)}")
            return False
        except AttributeError:
            self.errors.append(f"{api_name} does not exist in {module_name}.")
            return False
        except Exception as e:
            self.errors.append(str(e))
            return False

    def check_files(self, file_path, output_file_path):
        """
        Check the API calls in a list of Python files.
        """
        for dirpath, dirname, files in os.walk(file_path):
            for file_name in files:
                print('file name:', file_name)
                print('joined path:', os.path.join(dirpath, file_name))
                joined_path = os.path.join(dirpath, file_name)
                rel_path = os.path.relpath(joined_path, './seeds/unverified_seeds/Cluster/')

                # Check if the directory exists
                if not os.path.exists(os.path.dirname(os.path.join(output_file_path, rel_path))):
                    # Create the directory if it does not exist
                    os.makedirs(os.path.dirname(os.path.join(output_file_path, rel_path)))

                try:
                    with open(joined_path, 'r') as file:
                        print('reading')
                        content = file.read()
                        api_calls = self.extract_api_calls(content)
                        
                        for api in api_calls:
                            self.validate_api(api)


                        with open(os.path.join(output_file_path, rel_path), 'w') as output_file:
                            output_file.write(content)
                
                except FileNotFoundError:
                    self.errors.append(f"File {file_path} not found.")
                    with open(f'{os.path.join(output_file_path, rel_path)}_bad', 'w') as output_file:
                        output_file.write(f"File {file_path} not found.")
                except JSONDecodeError as e:
                    self.errors.append(f"Error decoding JSON in {file_path}: {str(e)}")
                    with open(f'{os.path.join(output_file_path, rel_path)}_bad', 'w') as output_file:
                        output_file.write(f"Error decoding JSON in {file_path}: {str(e)}")
                except Exception as e:
                    self.errors.append(f"Error processing {file_path}: {str(e)}")
                    with open(f'{os.path.join(output_file_path, rel_path)}_bad', 'w') as output_file:
                        output_file.write(f"Error processing {file_path}: {str(e)}")


if __name__ == "__main__":
    checker = APIChecker()
    root_path = './seeds/unverified_seeds'
    output_path = './seeds/verified_seeds'

    for dirpath, dirname, files in os.walk(root_path):
        current_dir = dirpath
        for file in files:
            print('current dir:', current_dir)
            checker.check_files(current_dir, output_path)

            if checker.errors:
                print("Errors found:")
                for error in checker.errors:
                    print(error)
            else:
                print("All APIs are valid.")

