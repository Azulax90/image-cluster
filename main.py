
from        tensorflow.keras.applications.resnet50  import ResNet50, preprocess_input
from        sklearn.cluster                         import KMeans
from        PIL                                     import Image
from        shutil                                  import move
import      tensorflow                              as tf
import      numpy                                   as np
import os


class clus:
    def __init__(self):
        self.input = "images"
        self.output = "output"
        self.clusters = 2
        self.names = []
        self.images = []
        

    def extract(self,image):
        model = ResNet50(weights='imagenet', include_top=False)
        img = np.expand_dims(image, axis=0)
        img = preprocess_input(img)
        features = model.predict(img)
        return features.flatten()
    
    def worker(self):
                
        for f in os.listdir(self.input):
            img = Image.open(os.path.join(self.input, f)).resize((224, 224))
            self.images.append(self.extract(np.array(img)))
            self.names.append(f)

        if self.images:
            kmeans = KMeans(n_clusters=min(self.clusters, len(self.images)), random_state=0, n_init=10).fit(np.array(self.images))

            for i in range(min(self.clusters, len(self.images))):
                os.makedirs(os.path.join(self.output, f'data_{i}'), exist_ok=True)

            for f, label in zip(self.names, kmeans.labels_):
                move(os.path.join(self.input, f), os.path.join(self.output, f'cluster_{label}', f))
        else:
            print("No imgs found")
            
dch_on_top =  clus()
dch_on_top.worker()