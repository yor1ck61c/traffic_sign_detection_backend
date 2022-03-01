import paddlex as pdx
from paddlex.det import transforms


def predict_img(img_local_path, img_name):
    """
    工具函数:
    检测和识别图片中包含的交通标识
    :param:
    img_local_path: 图片的本地绝对路径
    img_name: 图片名称
    :return: 预测结果的图片地址
    """
    test_transforms = transforms.Compose([
        transforms.Normalize(),
        transforms.ResizeByShort(),
        transforms.Resize(1024),
        transforms.RandomDistort()
    ])
    model = pdx.load_model('E:/VSCodeWorkSpace/traffic_sign_recognize/output/PPYOLO_mobilenetv1/epoch_300')
    # 预测结果Str
    result = model.predict(img_local_path, transforms=test_transforms)
    # image (str): 原图文件路径。
    # result (str): 模型预测结果。
    # threshold(float): score阈值，将Box置信度低于该阈值的框过滤不进行可视化。默认0.5
    # save_dir(str): 可视化结果保存路径。若为None，则表示不保存，该函数将可视化的结果以np.ndarray的形式返回；
    # 若设为目录路径，则将可视化结果保存至该目录下。默认值为'./'
    pdx.det.visualize(img_local_path, result, threshold=0.5, save_dir='./output/ResNet50_vd_ssld')
    predict_img_path = "E:/VSCodeWorkSpace/traffic_sign_recognize/output/ResNet50_vd_ssld/visualize_" + img_name
    category = translate_result_to_chinese(result)
    data = {
        "predict_img_path": predict_img_path,
        "category": category
    }
    return data


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


img_local_path = "E:/VSCodeWorkSpace/traffic_sign_recognize/dataset/roadsign_voc/upload_img/road358.png"
img_name = "road358.png"
data = predict_img(img_local_path, img_name)
print(data)
