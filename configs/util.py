import json 

def write_json(data,json_path):
    '''data数据写成json
    input:
        data: 需要写入的数据
        json_path: json文件路径
    output:
        无
    '''
    with open(json_path,'w') as f:
        json.dump(data,f,indent=4)

def parse_json(json_path):
    '''读、解析json
    input:
        json_path: json文件路径
    output:
        解析完成的数据
    ''' 
    with open(json_path,'r') as f:
        cfg = json.load(f)
    return cfg