from consts import wordnet_file
# imagenet superclasses!
#  (these already contain multiple classes out of the box.
#   however, they do not contain all relevant subclasses.
#   therefore, we map additional subclasses to these existing superclasses)
# e.g. a class from 'dog_classes' will map to the superclass 'n02084071' (dog)
superclasses = {
    'n02084071': 'dog',
    'n01503061': 'bird',
    'n02120997': 'cat',
    'n02858304': 'boat',
    'n02075296': 'bear',
}

dog_classes = [
    'n02084071', 'n02092468', 'n02095727', 'n02096756', 'n02104523',
    'n02093056', 'n02099029', 'n02090827',  'n02103406', 'n02087122',
    'n02087551', 'n02099997', 'n02104882', 'n02088839', 'n02112826',
    'n02085374', 'n02115335', 'n02095412', 'n02111626', 'n02108672',
    'n02100399', 'n02107420', 'n02090475', 'n02113335', 'n02109811',
    'n02101108', 'n02095050', 'n02101861', 'n02086478', 'n02103841',
    'n02106966'
]

bear_classes = [
    'n02131653'
]

cat_classes = [
    'n02121808'
]

boat_classes = [
    'n03790230', 'n04128499', 'n04244997', 'n04128837', 'n02858304',
]

# we probably will not be using cars (hard to segment properly, due to the nature of the scenes in which they appear)
#car_classes = [
#    'n04170037', 'n02959942', 'n04490091', 'n02924116', 'n02924116',
#    'n02958343', 'n04520170'
#]

def build_imagenet_map() -> dict:
    f = open(wordnet_file, 'r')

    # mapping that takes class ID => superclass ID
    imagenet_class_to_label = dict()
    for line in f.readlines():
        if len(line) < 1:
            continue
        superclass_id, class_id = line.split(' ')
        
        label = None
        if superclass_id in dog_classes:
            label = 'dog'
        elif superclass_id in boat_classes:
            label = 'boat'
        elif superclass_id in bear_classes:
            label = 'bear'
        elif superclass_id in cat_classes:
            label = 'cat'
        
        imagenet_class_to_label[class_id.replace('\n', '')] = label
    return imagenet_class_to_label