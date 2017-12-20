from PIL import Image
from feature import NPDFeature
from sklearn.externals import joblib
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os

postive_img_path = "./datasets/original/face"
negative_img_path = "./datasets/original/nonface"
positive_feature_path = "./datasets/feature/positive.feat"
negative_feature_path = "./datasets/feature/negative.feat"
adb_model_path = "./model/adb.model"
report_path = "./report.txt"

def extract_feature(img_path, feature_path):
    assert os.path.exists(img_path),"image path is not exist"
    sum = 0   # 特征总数
    fds = []  # 特征列表
    for childDir in os.listdir(img_path):
        f = os.path.join(img_path, childDir)
        pil_im = Image.open(f).convert('L').resize((24, 24), Image.ANTIALIAS)
        im = np.array(pil_im)
        im_feat = NPDFeature(im).extract()
        fds.append(im_feat)
        sum += 1
    joblib.dump(fds, feature_path)
    print ("%d sample features are extracted and saved." %sum)
    return fds


def pre_process():
    if not os.path.exists(os.path.split(positive_feature_path)[0]):
        os.mkdir(os.path.split(positive_feature_path)[0])
    extract_feature(postive_img_path, positive_feature_path)
    extract_feature(negative_img_path, negative_feature_path)


def get_feat_labels():
    fds = []
    labels = []
    p_sum = 0
    n_sum = 0

    data = joblib.load(positive_feature_path)
    for fd in data:
        fds.append(fd)
        p_sum += 1
    for x in range(0, len(data)):
        labels.append(1)

    data = joblib.load(negative_feature_path)
    for fd in data:
        fds.append(fd)
        n_sum += 1
    for x in range(0, len(data)):
        labels.append(-1)
    
    print("total %d positive sample and %d negative sample" % (p_sum, n_sum))
    return fds, labels


def train(X_train, y_train):
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             n_weakers_limit=20)
    print("Training a AdaBoost Classifier.")
    clf.fit(X_train, y_train)
    # If model directories don't exist, create them
    if not os.path.isdir(os.path.split(adb_model_path)[0]):
        os.makedirs(os.path.split(adb_model_path)[0])
    clf.save(clf, adb_model_path)


def test(X_test, y_test):
    print("Test the AdaBoost Classifier.")
    clf = AdaBoostClassifier.load(adb_model_path)
    pre_y = clf.predict(X_test)
    target_names = ['face', 'non face']
    report = classification_report(y_test, pre_y, target_names=target_names)
    print(report)
    with open(report_path, "w") as f:
        f.write(report)
        f.close()
        print("report has been wrote into %s" % report_path)


if __name__ == "__main__":
    if not os.path.exists(positive_feature_path):
        pre_process()
        fds, labels = get_feat_labels()
    else:
        fds, labels = get_feat_labels()

    X_train, X_test, y_train, y_test = train_test_split(fds, labels, test_size=0.2)

    if not os.path.exists(adb_model_path):
        train(X_train, y_train)
        test(X_test, y_test)
    else:
        test(X_test, y_test)



