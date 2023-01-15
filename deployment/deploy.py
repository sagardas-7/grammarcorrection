import pickle
import unicodedata
import regex as re
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
import matplotlib
import random
import math
from tensorflow.keras.models import Model
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding,LSTM,Dense,Softmax

import streamlit as st


train = pd.read_csv("./DATA/BACKUP DATA/TRAIN AND VALIDATION/train.csv")

validation = pd.read_csv("./DATA/BACKUP DATA/TRAIN AND VALIDATION/validation.csv")





with open('./DATA/BACKUP DATA/tokenizer_wrong.pickle', 'rb') as handle:
    tknizer_wrng = pickle.load(handle)


with open('./DATA/BACKUP DATA/tokenizer_correct.pickle', 'rb') as handle:
    tknizer_corr = pickle.load(handle)


vocab_size_wrng = 65901
vocab_size_corr = 65713



embeddings_index = dict()
f = open('./DATA/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix_enc = np.zeros((vocab_size_wrng+1, 100))
for word, i in tknizer_wrng.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix_enc[i] = embedding_vector



embedding_matrix_dec = np.zeros((vocab_size_corr+1, 100))
for word, i in tknizer_corr.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix_dec[i] = embedding_vector


### 2.1 <font color='blue'>**Encoder**</font>

class Encoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns encoder-outputs,encoder_final_state_h,encoder_final_state_c
    '''

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length,**kwargs):

        super().__init__()
        self.vocab_size = inp_vocab_size
        self.embedding_dim = embedding_size
        self.input_length = input_length
        self.enc_units= lstm_size
        self.lstm_output, self.lstm_state_h,self.lstm_state_c = 0,0,0
        #Initialize Embedding layer
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_decoder", weights=[embedding_matrix_enc], trainable=False)
        # Intialize Encoder LSTM layer
        self.lstm = LSTM(self.enc_units, return_state=True, return_sequences=True, name="Encoder_LSTM")
        
    
    def call(self,input_sequence,states):
        '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
          returns -- encoder_output, last time step's hidden and cell state
        '''
        #print("ENCODER ==> INPUT SQUENCES SHAPE :",input_sequence.shape)
        input_embedd                           = self.embedding(input_sequence)
        #print("ENCODER ==> AFTER EMBEDDING THE INPUT SHAPE :",input_embedd.shape)
        self.lstm_output, self.lstm_state_h,self.lstm_state_c = self.lstm(input_embedd)
        return self.lstm_output, self.lstm_state_h,self.lstm_state_c

      

    
    def initialize_states(self,batch_size):
      '''
      Given a batch size it will return intial hidden state and intial cell state.
      If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
      '''
      self.lstm_state_h = np.zeros((batch_size,self.enc_units))
      self.lstm_state_c = np.zeros((batch_size,self.enc_units))


    def get_config(self):
      config = super(Encoder,self).get_config()
      config.update({
          'vocab_size': self.vocab_size,
          'embedding_dim':self.embedding_dim,
          'input_length':self.input_length,
          'enc_units':self.enc_units,
          'lstm_output':self.lstm_output,
          'lstm_state_h':self.lstm_state_h,
          'lstm_state_c':self.lstm_state_c,
          'embedding':self.embedding,
          'lstm':self.lstm,
      })
      return config


#defing class attention functions

class Attention ( tf.keras.layers.Layer ) :

    def __init__ ( self , scoring_function , att_units ) :

        super().__init__( )
        self.scoring_function  = scoring_function
        self.att_units =  att_units
        if self.scoring_function == 'dot' :
            pass
        elif scoring_function == 'general' :
            # Intialize variables for geraneal fun needed
            self.wa = Dense ( att_units )
        elif scoring_function == 'concat' :
            self.w1 = Dense ( att_units )
            self.w2 = Dense ( att_units ) 
            self.v = Dense ( 1 )

    #defining call function
    def call ( self , decoder_hidden_state , encoder_output ):

        if self.scoring_function == 'dot' :
            state = tf.expand_dims ( decoder_hidden_state , -1 )
            score = tf.matmul ( encoder_output , state )
            weights  = tf.nn.softmax ( score , axis = 1 )
            weighted_out =  encoder_output * weights
            context_vec =  tf.reduce_sum ( weighted_out , axis = 1 )
            #returing its weights
            # #print(context_vec)
            # #print(weights)
            return context_vec , weights
        
        elif self.scoring_function == 'general' :
            state = tf.expand_dims ( decoder_hidden_state , 2 )                                    
            score = tf.matmul ( self.wa (encoder_output ) , state )                        
            weights = tf.nn.softmax ( score , axis = 1 )  
            weighted_out = encoder_output * weights
            context_vec = tf.reduce_sum ( weighted_out , axis = 1 )
            #return contextvec and weights           
            return context_vec , weights

        elif self.scoring_function  == 'concat' :         
            state = tf.expand_dims ( decoder_hidden_state , 1 )           
            score = self.v ( tf.nn.tanh ( self.w1 ( state ) + self.w2 ( encoder_output ) ) )
            weights = tf.nn.softmax ( score , axis = 1 )
            #weighted output
            weighted_out = encoder_output * weights            
            context_vec = tf.reduce_sum ( weighted_out , axis = 1 )            
            #returing weights       
            return context_vec , weights 


    def get_config(self):
      config = super(Attention,self).get_config()
      config.update({
          'scoring_function': self.scoring_function,
          'att_units':self.att_units,
          'wa':self.wa,
          'w1':self.w1,
          'w2':self.w2,
          'v':self.v
      })
      return config


### 2.3 <font color='blue'>**OneStepDecoder**</font>

class One_Step_Decoder(tf.keras.Model):
  def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
      # Initialize decoder embedding layer, LSTM and any other objects needed
      super().__init__()
      self.vocab_size = tar_vocab_size
      self.embedding_dim = embedding_dim
      self.input_length = input_length
      self.dec_units = dec_units
      self.score_fun = score_fun
      self.att_units = att_units
      #Initialize Embedding layer
      self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_decoder", weights=[embedding_matrix_dec], trainable=False)
      #Intialize Decoder LSTM layer
      self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True, name="Encoder_LSTM") 
      #Intialize Dense layer(out_vocab_size) with activation='softmax'
      self.dense   = Dense(self.vocab_size)
      # self.dense   = Dense(self.vocab_size, activation='softmax')
      self.attention                                                         = Attention(self.score_fun,self.att_units)
      self.out_temp = []
      self.decoder_final_state_h,self.decoder_final_state_c = [],[]
      self.context_vector,self.attention_weights = [],[]
      self.concat = []
      



  def call(self,input_to_decoder, encoder_output, state_h,state_c):
    '''
        One step decoder mechanisim step by step:
      A. Pass the input_to_decoder to the embedding layer and then get the output(batch_size,1,embedding_dim)
      B. Using the encoder_output and decoder hidden state, compute the context vector.
      C. Concat the context vector with the step A output
      D. Pass the Step-C output to LSTM/GRU and get the decoder output and states(hidden and cell state)
      E. Pass the decoder output to dense layer(vocab size) and store the result into output.
      F. Return the states from step D, output from Step E, attention weights from Step -B
    '''
    target_embedd                                                                   = self.embedding(input_to_decoder)
    self.context_vector,self.attention_weights                                      = self.attention(state_h,encoder_output)

    self.concat = tf.concat([tf.expand_dims(self.context_vector,1),target_embedd],axis=-1)
    initial_states                                                                  = [state_h,state_c]
    decoder_output,self.decoder_final_state_h,self.decoder_final_state_c            = self.lstm(self.concat, initial_state=initial_states)
    # #print("LSTM OUTPUT-->",decoder_output[:5])
    # decoder_output = tf.reshape(decoder_output,[tf.shape(decoder_output).numpy()[0],tf.shape(decoder_output).numpy()[2]])
    decoder_output = tf.reshape(decoder_output,(-1,decoder_output.shape[2]))
    output                                                                          = self.dense(decoder_output)
    # #print("DENSE OUTPUT-->",output[:5])
    # #print(decoder_output.shape)
    # #print(output.shape)


    return output,self.decoder_final_state_h,self.decoder_final_state_c,self.attention_weights,self.context_vector
    # return np.array(output),self.decoder_final_state_h,self.decoder_final_state_c,self.attention_weights,self.context_vector

  
  def get_config(self):
      config = super(One_Step_Decoder,self).get_config()
      config.update({
          'vocab_size': self.vocab_size,
          'embedding_dim':self.embedding_dim,
          'input_length':self.input_length,
          'dec_units':self.dec_units,
          'score_fun':self.score_fun,
          'att_units':self.att_units,
          'embedding':self.embedding,
          'lstm':self.lstm,
          'dense':self.dense,
          'attention':self.attention,
          'out_temp':self.out_temp,
          'decoder_final_state_h':self.decoder_final_state_h,
          'decoder_final_state_c':self.decoder_final_state_c,
          'context_vector':self.context_vector,
          'attention_weights':self.attention_weights,
          'concat':self.concat,
      })
      return config


### 2.4 <font color='blue'>**Decoder**</font>

class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
      #Intialize necessary variables and create an object from the class onestepdecoder
      super().__init__()
      self.vocab_size = out_vocab_size
      self.embedding_dim = embedding_dim
      self.input_length = input_length
      self.dec_units = dec_units
      self.score_fun = score_fun
      self.att_units = att_units
      #Initialize Embedding layer
      self.onestep_decoder = One_Step_Decoder(self.vocab_size, self.embedding_dim, self.input_length, self.dec_units ,self.score_fun ,self.att_units)
      self.all_outputs = tf.TensorArray(tf.float32,size=2,name="output_arrays")

      


    
    def call(self, input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state ):

        #Initialize an empty Tensor array, that will store the outputs at each and every time step
        self.all_outputs = tf.TensorArray(tf.float32,size=tf.shape(input_to_decoder)[1],name="output_arrays")
      
        #Iterate till the length of the decoder input
        for timestep in range(tf.shape (input_to_decoder)[1]):
            # Call onestepdecoder for each token in decoder_input
            output,decoder_hidden_state,decoder_cell_state,_,_ = self.onestep_decoder(input_to_decoder[:,timestep:timestep+1],encoder_output,decoder_hidden_state,decoder_cell_state)
            # Store the output in tensorarray
            self.all_outputs = self.all_outputs.write(timestep,output)
        # Return the tensor array.
        # #print("self.all_outputs shape-->",self.all_outputs.size())
        # #print("self.all_outputs-->",self.all_outputs)
        self.all_outputs = tf.transpose(self.all_outputs.stack(),[1,0,2])
        return self.all_outputs

    
    def get_config(self):
      config = super(One_Step_Decoder,self).get_config()
      config.update({
          'vocab_size': self.vocab_size,
          'embedding_dim':self.embedding_dim,
          'input_length':self.input_length,
          'dec_units':self.dec_units,
          'score_fun':self.score_fun,
          'att_units':self.att_units,
          'onestep_decoder':self.onestep_decoder,
          'all_outputs':self.all_outputs
      })
      return config
        
        
### 2.5 <font color='blue'>**Encoder Decoder model**</font>

max_length = 70
enc_units = 256
embedding_dim = 100

class encoder_decoder(tf.keras.Model):
  def __init__(self,encoder_inputs_length,decoder_inputs_length, output_vocab_size,score_fun):
    #Intialize objects from encoder decoder
    super().__init__()
    self.score_fun = score_fun
    self.encoder = Encoder(inp_vocab_size=vocab_size_wrng + 1, embedding_size=embedding_dim, input_length=encoder_inputs_length, lstm_size=enc_units)#https://stackoverflow.com/questions/48479915/what-is-the-preferred-ratio-between-the-vocabulary-size-and-embedding-dimension
    self.decoder = Decoder(out_vocab_size=vocab_size_corr + 1, embedding_dim=embedding_dim, input_length=decoder_inputs_length, dec_units=enc_units,score_fun=self.score_fun,att_units=enc_units)
    self.decoder_output = []
  
  
  def call(self,data):

    input,output = data[0],data[1]
    #Intialize encoder states, Pass the encoder_sequence to the embedding layer
    #print("="*20, "ENCODER", "="*20)
    batch_size = 16
    enc_initial_state = self.encoder.initialize_states(batch_size)
    encoder_output, encoder_h, encoder_c = self.encoder(input,enc_initial_state)
    #print("ENCODER ==> OUTPUT SHAPE",encoder_output.shape)
    #print("ENCODER ==> HIDDEN STATE SHAPE",encoder_h.shape)
    # #print("ENCODER ==> CELL STATE SHAPE", encoder_c.shape)
    # #print("="*20, "DECODER", "="*20)
    # Decoder initial states are encoder final states, Initialize it accordingly
    dec_initial_state = [encoder_h,encoder_c]
    # Pass the decoder sequence,encoder_output,decoder states to Decoder
    self.decoder_output                             = self.decoder(output, encoder_output,encoder_h,encoder_c)
    # return the decoder output
    return self.decoder_output


  def get_config(self):
    config = super(One_Step_Decoder,self).get_config()
    config.update({
        'score_fun':self.score_fun,
        'encoder':self.encoder,
        'decoder':self.decoder,
        'decoder_output':self.decoder_output
    })
    return config


### 2.6 <font color='blue'>**Custom loss function**</font>

#https://www.tensorflow.org/tutorials/text/image_captioning#model
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    """ Custom loss function that will not consider the loss for padded zeros.
    why are we using this, can't we use simple sparse categorical crossentropy?
    Yes, you can use simple sparse categorical crossentropy as loss like we did in task-1. But in this loss function we are ignoring the loss
    for the padded zeros. i.e when the input is zero then we donot need to worry what the output is. This padded zeros are added from our end
    during preprocessing to make equal length for all the sentences.

    """
    
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

### 2.7 <font color='blue'>Creating data pipeline

class Dataset:
    def __init__(self, data, tknizer_ita, tknizer_eng, max_len):
        self.encoder_inps = data['wrong'].values
        self.decoder_inps = data['corr_inp'].values
        self.decoder_outs = data['corr_out'].values
        self.wrng_tokenizer = tknizer_wrng
        self.corr_tokenizer = tknizer_corr
        self.max_len = max_len

    def __getitem__(self, i):
        self.encoder_seq = self.wrng_tokenizer.texts_to_sequences([self.encoder_inps[i]]) # need to pass list of values
        self.decoder_inp_seq = self.corr_tokenizer.texts_to_sequences([self.decoder_inps[i]])
        self.decoder_out_seq = self.corr_tokenizer.texts_to_sequences([self.decoder_outs[i]])

        self.encoder_seq = pad_sequences(self.encoder_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_inp_seq = pad_sequences(self.decoder_inp_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_out_seq = pad_sequences(self.decoder_out_seq, maxlen=self.max_len, dtype='int32', padding='post')
        return self.encoder_seq, self.decoder_inp_seq, self.decoder_out_seq

    def __len__(self): # your model.fit_gen requires this function
        return len(self.encoder_inps)

    
class Dataloder(tf.keras.utils.Sequence):    
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.dataset.encoder_inps))


    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.squeeze(np.stack(samples, axis=1), axis=0) for samples in zip(*data)]
        # we are creating data like ([italian, english_inp], english_out) these are already converted into seq
        return tuple([[batch[0],batch[1]],batch[2]])

    def __len__(self):  # your model.fit_gen requires this function
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)


def load():
    #print("INSIDE LOAD!!!!!!!!!!!!!!\n\n\n\n\n\n\n\n\n\n\n")
    train_dataset = Dataset(train[:2], tknizer_wrng, tknizer_corr, max_length)
    test_dataset  = Dataset(validation[:2], tknizer_wrng, tknizer_corr, max_length)


    train_dataloader = Dataloder(train_dataset, batch_size=2)
    # test_dataloader = Dataloder(test_dataset, batch_size=2)


    # #print(train_dataloader[0][0][0].shape, train_dataloader[0][0][1].shape, train_dataloader[0][1].shape)
    # #print(test_dataloader[0][0][0].shape, test_dataloader[0][0][1].shape, test_dataloader[0][1].shape)

    # full_model4 = encoder_decoder(encoder_inputs_length=max_length,decoder_inputs_length=max_length,output_vocab_size=vocab_size_corr,score_fun='dot')
    # optimizer = tf.keras.optimizers.Adam(0.001,clipnorm=0.001)
    # full_model4.compile(optimizer = optimizer,loss=loss_function)

    # history = full_model4.fit(train_dataloader, epochs=25, validation_data=test_dataloader,callbacks = callbacks)
    # full_model4.summary()

    # full_model4.save_weights("/content/drive/MyDrive/DATA SCIENCE/CASE STUDY 2/MODELS/FINAL_ATTENTION_BLEU_81.h5")

    ### 2.9 <font color='blue'>Loading the weights

    dummy = encoder_decoder(encoder_inputs_length=max_length,decoder_inputs_length=max_length,output_vocab_size=vocab_size_corr,score_fun='dot')
    optimizer = tf.keras.optimizers.Adam()
    dummy.compile(optimizer = optimizer,loss=loss_function)
    dummy.fit(train_dataloader, epochs=1)



    dummy.load_weights('./DATA/FINAL_ATTENTION_BLEU_49.h5')
    return dummy








def predict(dummy,input_sentence):

  '''
  A. Given input sentence, convert the sentence into integers using tokenizer used earlier
  B. Pass the input_sequence to encoder. we get encoder_outputs, last time step hidden and cell state
  C. Initialize index of <start> as input to decoder. and encoder final states as input_states to decoder
  D. till we reach max_length of decoder or till the model predicted word <end>:
         predicted_out,state_h,state_c=model.layers[1](dec_input,states)
         pass the predicted_out to the dense layer
         update the states=[state_h,state_c]
         And get the index of the word with maximum probability of the dense layer output, using the tokenizer(word index) get the word and then store it in a string.
         Update the input_to_decoder with current predictions
  F. Return the predicted sentence
  '''
  input_sentence = input_sentence.replace("'", '')
  temp_token = tknizer_wrng.texts_to_sequences ( [ input_sentence ] )
  temp_token  = pad_sequences ( temp_token , maxlen = vocab_size_wrng , dtype = 'int32' , padding = 'post' )



  #print("=" * 30, "Inference", "=" * 30)
  initial_state =  [np.zeros((1,64)) ,np.zeros((1,64))]
  enc_output, enc_state_h, enc_state_c = dummy.layers[0](np.expand_dims(temp_token[0], 0),initial_state)
  states_values = [enc_state_h, enc_state_c]

  dec_inp = np.array ( tknizer_corr.word_index ['<start>'] ).reshape( 1 , 1 )
  attention_weights_list = []


  predicted_eng = ""
  for i in range(max_length):

    predictions , enc_state_h , enc_state_c , attention_weights , context_vector = dummy.layers[ 1 ].onestep_decoder( dec_inp, enc_output , enc_state_h, enc_state_c )
    
    word_ind  = np.argmax ( predictions,-1 )
    pred_str = list ( tknizer_corr.word_index.keys( ) ) [ int ( word_ind - 1 ) ]

    attention_weights_list.append ( attention_weights [0,:,0 ] )

    predicted_eng+= pred_str + " "

    dec_inp = word_ind.reshape(1,1)

    if(pred_str == "<end>"):
      # #print(predicted_eng)
      return predicted_eng, np.array ( attention_weights_list )
  return predicted_eng, np.array ( attention_weights_list )




def plot_attention ( attention , sentence , predicted_sentence ) :
       
    sentence  = sentence.split()
    #final sentence declaration
    sentence  = sentence 
    
    predicted_sentence =  predicted_sentence.split() + [ '<end>' ]    
    fig = plt.figure(figsize =( 10 , 10 ))
    ax = fig.add_subplot (1 , 1 , 1)
    attention = attention [:len ( predicted_sentence ), :len(sentence) ]
    
    #matrix plot with proper arguments
    ax.matshow(attention, cmap = 'viridis', vmin = 0.0)

    
    #fontsize as 14
    fontdict = {'fontsize': 14}
    #seting up axis labels argument
    ax.set_xticklabels( [''] + sentence , fontdict = fontdict , rotation = 90 )
    ax.set_yticklabels( [''] + predicted_sentence , fontdict = fontdict)
    ax.xaxis.set_major_locator ( matplotlib.ticker.MultipleLocator (1) )
    ax.yaxis.set_major_locator ( matplotlib.ticker.MultipleLocator (1) )
    
    ax.set_xlabel ( 'Input text' )   
    ax.set_ylabel ( 'Output text' )
    
    #titles
    plt.suptitle ('Attention weights')



def main():
    st.title("GRAMMAR CORRECTION")
    model = load()
    input = st.text_input("INPUT STRING")
    pred_sent=''
    if st.button("CORRECT THE SENTENCE!!"):
        pred_sent,attention_weights   = predict (model,input)
        pred_sent = pred_sent.replace("<end>","")
        st.success(pred_sent)



if __name__=="__main__":

    main()