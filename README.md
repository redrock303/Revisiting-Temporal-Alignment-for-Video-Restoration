 # Revisiting Temporal Alignment for Video Restoration CVPR-2022 <br>
#### Kun Zhou, Wenbo Li, Liying Lu, Xiaoguang Han, Jiangbo Lu
#### [\[paper\]](https://arxiv.org/pdf/2111.15288v2.pdf) <br>
---
#### We have provided the source code of our video super-resolution, video deblurring, and video denoising models. <br>
#### We provide our results at [Google Cloud](https://drive.google.com/drive/folders/1EMWTJhRXR6F3-6Mk-4T09kB5qSMcs1iS?usp=sharing). <br>
#### The pre-trained models are uploaded in the [google cloud](https://drive.google.com/drive/folders/1_em2Z1gUe9K3rbEFFvVENUcL4cY22Tkq). <br>
#### Some in-the-wild testing sequences are available [here](https://drive.google.com/drive/folders/1c9X3UlmoS7xkgTfPC4IK9iBInEW1V-l8). <br>
---
File Structure  <br>
   >libs <br>
   >>DcNv2 <br>
   
   >utils <br>
   >>common.py  <br>
   >>core.py  <br>
   >>model_opr.py  <br>
   
   >models  <br>
   >>VDB  <br>
   >>>config.py   <br>
   >>>network.py  <br>
   >>>validate.py  <br>
   >>>sequence_test.py  <br>
   >>>load_VDB_Data.py  <br>
   >>>VideoDeblur.py  <br>
   
   >>VDN  <br>
  >>>config.py   <br>
  >>>network.py  <br>
  >>>validate.py  <br>
  >>>validate_davis.py  <br>
  >>>sequence_test.py  <br>
  
  >>VSR_REDS  <br>
  >>>config.py   <br>
  >>>network.py  <br>
  >>>validate.py  <br>
  
   >>VSR_VIMEO90K  <br>
   >>>config.py   <br>
   >>>network.py  <br>
   >>>validate.py  <br>
   >>>sequence_test.py  <br>
---
## Usage
The DCNv2 should be installed correctly by running: <br>
      mask.sh in ./libs/DCNv2_latest/ <br>
For evaluating the results of each model, you can run the corresponding "validate.py". <br>
Also you can run the sequence_test.py for testing your own video sequences. <br>

## Citing
#### If you find this code useful for your research, please consider citing the following paper:
    @inproceedings{zhou2021rta, 
      title={Revisiting Temporal Alignment for Video Restoration},
      author={Kun Zhou and Wenbo Li and Liying Lu and Xiaoguang Han and Jiangbo Lu}, 
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition} 
      year={2022} 
    } 
---
## License 
Our code is for research purposes only. 

