# Revisiting Temporal Alignment for Video Restoration CVPR-2022
## Kun Zhou, Wenbo Li, Liying Lu, Xiaoguang Han, Jiangbo Lu
[\[paper\]](https://arxiv.org/pdf/2111.15288v2.pdf) <br>
---
We have provided the source code of our video super-resolution, video deblurring, and video denoising models. The validation codes are also given in the corresponding folders. For in-the-wild data, the sequence_test.py scripts in each folder can be used for testing. 
We provide our results at [Google Cloud](https://drive.google.com/drive/folders/1EMWTJhRXR6F3-6Mk-4T09kB5qSMcs1iS?usp=sharing).
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
The DCNv2 should be installed correctly by running:
      mask.sh in ./libs/DCNv2_latest/
For evaluating the results of each model, you can run the corresponding "validate.py".
Also you can run the sequence_test.py for testing your own video sequences.

### Citing
If you find this code useful for your research, please consider citing the following paper:

@inproceedings{zhou2021rta, <br>
      title={Revisiting Temporal Alignment for Video Restoration},<br>
      author={Kun Zhou and Wenbo Li and Liying Lu and Xiaoguang Han and Jiangbo Lu},<br>
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition}
      year={2021}<br>
    }<br>
