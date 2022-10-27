import streamlit as st
# from googletrans import Translator
# import numpy as np
# from clean import clean_font
from transformers import AutoTokenizer, MBartForConditionalGeneration, AutoModel
from transformers.pipelines import pipeline
# from vncorenlp import VnCoreNLP
# vnp=VnCoreNLP("vncore/VnCoreNLP-1.1.1.jar",annotators="wseg")

# Tạo hàm xử lý wordsegment
# def wsegm(st):

    # vn_wsegm=[]
    # a=vnp.tokenize(st)
    # # b=' '.join(a[0])
    # return b
  

# 
# 
# translate=Translator()
tokenizer = AutoTokenizer.from_pretrained("mrgreat1110/FunixTranslation_tokenizer", use_fast=True)
model = MBartForConditionalGeneration.from_pretrained("mrgreat1110/FunixTranslation_model")
st.header("This demo version for FunixXseries Machine Translation")
question = st.text_area('Insert a English sentences.')
# button=st.button('Translate')
# if question:
    # vi=translate.translate(str(question), src='en', dest='vi').text
    # TXT = vi
    # TXT=clean_font(TXT)
    # TXT=wsegm(TXT)
    # 
    
input_ids = tokenizer([question], return_tensors="pt")
predict=model(input_ids['input_ids'])
logits=predict.logits
    # probs = tf.nn.softmax(logits[0].detach().numpy())
    # test=np.argmax(probs, axis=1)
    # ans=tokenizer.decode(test)
    
    # 
    # st.write(TXT)
