from deepface import DeepFace
from os import path
from glob import glob
from random import choice

p_db = "/media/manjunath/Data/Datasets/Face-Recognition/lfw-sample"
p_person_dirs = [ x for x in glob(path.join(p_db, "*")) if path.isdir(x) ]

for p_person in p_person_dirs:
    p_imgs = glob(path.join(p_person, "*.jpg"))
    p_rand_img = choice(p_imgs)

    print(f"Person {path.basename(p_person)}")
    print(f"Rand Img {path.basename(p_rand_img)}")

    try:
        dfs = DeepFace.find( img_path = p_rand_img, db_path = p_db  )
        for df in dfs:
            print(df)
    except ValueError as e:
        print("Error. Face not found")
        print(e)
