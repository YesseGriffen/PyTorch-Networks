# PyTorch-Networks
Neural Networks made in PyTorch
<pre>
How to use <br/>
--------------------------- <br/>
1. MainAction Folder = Creates the Dataset, Actions used for the Model, and Main File for running a Model. <br/>
                     = Later on there will be sub-folders for different use-cases, maybe for different problems, more features in the ActionEngine, etc. <br/>
                     = This is the base code that helps run the Model File that is chosen and sets up a test environment in MainEngine to use it. <br/>

2. Models Folder     = Every model will be stored in here, The name of the file will be what the model is based around,  <br/> 
                     = Later on there will be more models for different use-cases. <br/>
                     = EX: LSTM/Attention, EX: EncoderTransformer, EX: GRNTransformer, etc. <br/>
<br/>
<br/>

Example use case
--------------------------- <br/>
Requirements --- torch == 2.6.0+cu126 *Don't need cuda, regular torch will work, but is slower*
             --- matplotlib == 3.10.1 *Optional, if you don't want to use graphs comment out graph_results in ActionEngine as well as the call in MainEngine*

Pick a model from the models folder.

In a code editor make sure ActionEngine is importing the right file.

Then you should be able to run with all the files in the same folder.
  
  
</pre>
