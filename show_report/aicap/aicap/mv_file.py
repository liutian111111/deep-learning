import os
import sys
import random
import math

def mv_file():
#if __name__=="__main__":
    base_dir='/home/gpu/tensorflow/tensorflow/ai-cap/'
    pre_dir=base_dir+'predict_list/'
    test_dir=base_dir+'pro_cap/captcha/'
    static_dir=base_dir+'show_report/aicap/template/static/'
    pre_imgs=os.listdir(pre_dir)
    test_imgs=os.listdir(test_dir)
    static_imgs=os.listdir(static_dir)
    if(len(pre_imgs)==1):
	os.remove(pre_dir+pre_imgs[0])
    if(len(static_imgs)==1):
	os.remove(static_dir+static_imgs[0])
    test_num=(int)(random.uniform(0,len(test_imgs)))
    print(test_num)
    test_file=test_imgs[test_num]
    s_targetfile=static_dir+test_file
    p_targetfile=pre_dir+test_file
    sourcefile=test_dir+test_file
    #source_fid=open(sourcefile,"rb")
    open(s_targetfile, "wb").write(open(sourcefile,'rb').read())
    open(p_targetfile,'wb').write(open(sourcefile,'rb').read())
    
    
