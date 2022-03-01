import paddlex as pdx
import json
from paddlex.det import transforms
from flask import Flask, request, jsonify

my_app = Flask(__name__)  # 实例化
my_app.config['JSON_AS_ASCII'] = False


def get_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    @param
    img_local_path:文件单张图片的本地绝对路径
    @return: 图片流
    """
    import base64
    with open(img_local_path, 'rb') as local_img:
        img_stream = local_img.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


def translate_result_to_chinese(result):
    """
    将模型预测结果中的英文转换为中文
    :param result: 模型的预测结果
    :return: 中文版预测结果
    """
    english_result = result[0]["category"]
    if english_result == "speedlimit":
        return "限速"
    return "unknown"


def predict_img(img_local_path, img_name):
    """
    识别算法:
    检测和识别图片中包含的交通标识
    @param:
    img_local_path: 图片的本地绝对路径
    img_name: 图片名称
    @return: 预测结果的图片地址和中文分类
    """
    test_transforms = transforms.Compose([
        transforms.Normalize(),
        transforms.ResizeByShort(),
        transforms.Resize(1024),
        transforms.RandomDistort()
    ])
    model = pdx.load_model('C:/Users/ASUS/PycharmProjects/pythonProject/output/epoch_300')
    result = model.predict(img_local_path, transforms=test_transforms)
    pdx.det.visualize(img_local_path, result, threshold=0.5, save_dir='./output/ResNet50_vd_ssld')
    predict_img_path = "C:/Users/ASUS/PycharmProjects/pythonProject/output/ResNet50_vd_ssld/visualize_" + img_name
    category = translate_result_to_chinese(result)
    data = {
        "predict_img_path": predict_img_path,
        "category": category
    }
    return data


@my_app.route("/upload", methods=['POST'])
def main_function():
    # 获取前端图片
    img = request.files.get('img')
    base_dir = "C:/Users/ASUS/PycharmProjects/pythonProject/upload_img/"
    img_name = img.filename
    # 拼接待预测图片的地址，并将图片保存
    img_local_path = base_dir + img_name
    img.save(img_local_path)
    # 对刚刚保存的图片进行预测
    data = predict_img(img_local_path, img_name)
    data["origin_img_path"] = get_img_stream(img_local_path)
    # 对预测图片进行base64编码，并转成字符串返回
    data["predict_img_path"] = get_img_stream(data["predict_img_path"])
    return jsonify(data)


if __name__ == '__main__':
    # 主函数
    # app.run(host, port, debug, options)
    # 默认值：host="127.0.0.1", port=5000, debug=False
    my_app.run(host="127.0.0.1", port=5000)
