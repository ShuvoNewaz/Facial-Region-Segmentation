# !/bin/bash
data_dir=dataset
pretrained_dir=pretrained
saved_model_dir=saved_model
saved_resnet_dir="${pretrained_dir}/resnet"
saved_pspnet_dir="${saved_model_dir}/pspnet"
# Change the following as necessary; you may not be Shuvo.
haar_cascade_dir="/home/shuvo/anaconda3/envs/face_segmentation/lib/python3.12/site-packages/cv2/data"

for dir in "${data_dir}" "${pretrained_dir}" "${haar_cascade_dir}"\
            "${saved_resnet_dir}" "${saved_pspnet_dir}";
do
    mkdir -p "${dir}"
done

# Download and organize dataset

data_ref=ashish2001/multiclass-face-segmentation
kaggle datasets download -d "${data_ref}"
unzip multiclass-face-segmentation.zip -d "${data_dir}"
rm multiclass-face-segmentation.zip

declare -a splits=("train" "val")
split_parent="${data_dir}/content/All_data"
for split in "${splits[@]}";
do
    mv "${split_parent}/${split}" "${data_dir}/${split}"
done
rm -rf "${data_dir}/content"

# Download and organize model weights

## PSPNet
gdown --fuzzy https://drive.google.com/file/d/1_C2OwZ3jztEQoc7gUbbaDEDIFLUZuzHx/view?usp=drive_link
mv face_weights.pt "${saved_pspnet_dir}/checkpoint.pt"

##ResNet
declare -A resnet_links
resnet_links["resnet18"]="https://download.pytorch.org/models/resnet18-5c106cde.pth"
resnet_links["resnet50"]="https://download.pytorch.org/models/resnet50-19c8e357.pth"
resnet_links["resnet101"]="https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
resnet_links["resnet152"]="https://download.pytorch.org/models/resnet152-b121ed2d.pth"

declare -A resnet_local_names
resnet_local_names["resnet18"]="resnet18.pth"
resnet_local_names["resnet50"]="resnet50_v2.pth"
resnet_local_names["resnet101"]="resnet101_v2.pth"
resnet_local_names["resnet152"]="resnet152_v2.pth"

for resnet in "${!resnet_links[@]}";
do
    wget -O "${saved_resnet_dir}/${resnet_local_names[$resnet]}"\
    "${resnet_links[${resnet}]}"
done

# Download Haar Cascades

haar_cascade_homepage=https://github.com/opencv/opencv/raw/refs/heads/master/data/haarcascades
declare -a haar_cascade_downloads=("haarcascade_eye.xml"
                                    "haarcascade_profileface.xml"
                                    "haarcascade_frontalface_default.xml")
for haar_cascade in "${haar_cascade_downloads[@]}";
do
    wget -O "${haar_cascade_dir}/${haar_cascade}" "${haar_cascade_homepage}/${haar_cascade}"
done