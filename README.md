# Muti-modal-Creative-Story-Generation-from-Images-using-captioning
In this project the stories are generated using a two part framework as descriptive image captioning using smaller NLP models and then using a large language model for generating creative stories using those. The aim of the work is to generate stories which are coherent with the images and can incorporate the theme and context presented. At the end of the project the results look like the follwing: For pairs of three images, one creative story is generated using first the Image Captioning model which takes images as the input and then the Story Generation model which takes the captions as the input. 

![Image Captioning Model](images/story.png?raw=true)

## 1. Dataset 

Different daatses are used for the two different tasks and the captioning model is tested on a separate dataset than it is trained on.
### Image Captioning Datasets
#### MS COCO: 
A large dataset with 328,000 images and 5 captions per image, used for training both the baseline (Neural) and CLIP Prefix models for image captioning. The Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) dataset is a large-scale dataset generally used for scene understanding.
##### Install COCO API

1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)
  * 
#### Flickr8k:
A smaller dataset with 8,000 images, each associated with five different captions, used as the test dataset.

### Story Generation Dataset
#### Writing Prompts: 
Consists of 300,000 human-written stories paired with writing prompts, used for Fine-tuning the GPT2 model for Story Generation. 

### Preprocessing
For image captioning, The MS COCO dataset was further split and pre-processed into train and validate datasets and the images, their associated captions were used to extract features (VGG features for the baseline model), to train the model for image captioning. Data preprocessing for the baseline model involves converting text to lowercase, removing special characters and numbers, and tokenizing the sentences. Word embeddings are generated using an embeddings layer. Data is generated in batches to manage resource consumption effectively. 

For story generation, the dataset is divided into prompts and stories, both of which are utilized to train the model to generate creative and coherent text. During the fine-tuning process, the train data consisted of sequence of sentences in the format: <prompt + ’ <sep> ’ + story>

## Structure of the Experiments
The entire work and all the different experiments including the Baselines as well as the implemetation of the final code has been in the form of five Colab notebooks. 

A neural network architecture consisting of both CNNs (Encoder) and Decoder with attention mechanism is used to automatically generate captions from images as the baseline model. For story generation, the baseline model used is a fine-tuned GPT-2 model which generates creative captions based on the prompts given (image captions along with initial prompt).

## Training Baseline Models 
The baselines models are implement in the notebooks [Image Captioning on COCO Dataset](https://github.com/kashishgoel9/Muti-modal-Creative-Story-Generation-from-Images-using-captioning/blob/main/Baseline_BLUE_image_captioning_on_coco_dataset.ipynb) and the [Story Generation GPT-2 Baseline](Story Genration_Fine-Tuned-GPT2_baseline.ipynb)

### Image Captioning Model (Neural Model)
The baseline implemented image captioning model employs an encoder-decoder architecture, consisting of a pre-trained InceptionV3 convolutional neural network (CNN) with Multi-head Attention Transformer layer as the encoder to process images. The transformer encoder layer is designed to process the image features, utilizing attention mechanisms and layer normalization for efficient learning. The encoder processes image data and generates a feature vector, while the decoder generates captions for the images based on the encoded information. 

For the text generation part, a transformer decoder layer is used. It incorporates embeddings that combine token and position information, and employs multi-head attention mechanisms to focus on different parts of the input sequence. The decoder also includes feed-forward networks and dropout layers for regularization.
The model is trained to minimize loss and maximize accuracy, with the ability to augment images during training for better generalization.

### Story Generation Model (With and W/o Fine-tuned GPT-2 Model)
The implemented model is GPT-2, which involves first zero-shot transfer learning, where the model is pre-trained on language modeling without task-specific fine tuning first. 

The model is then fine-tuned using a dataset of writing prompts and stories, and then it's used to generate stories based on these prompts. 
Data preparation involves using the validation dataset as the training dataset due to time constraints. The pretrained GPT-2 model is readily available through the transformers package. 

## Training Experiments
For the final pipeline implementation, there were further experiments conducted and different better models were tried and the best models were chosen to be integrated to create the image-to-story system. The two notebooks contain the code for the fine-tuning other models for Story Generation: [Fien-tuning BART model for Story](Story Generation_Fine-Tuning_BART.ipynb) and the [Fine-tuning Flan-t5 Model for Story](Story Generation_Fine-Tune_Flan-T5.ipynb)

Primarily three experiments were conducted to test whether story generation and image captioning models are able to perform well for the proposed tasks. The two experiments performed are conducted individually for the two models, while the third experiment evaluates the performance of the entire system.  

### CLIP Prefix Model:
The CLIP Prefix model is implemented using CLIP embeddings and pre-trained CLIP model along with GPT-2. The training parameters and the hyperparameters used are mentioned in the notebook [Image-to-Story System](CLIP Prefix_Zero-shot_Fine-tuned GPT-2_Img-to-Story.ipynb)
First, I evaluated the Image Captioning model by evaluating the quality of captions generated by the model. For this, the captions generated are evaluated using the BLUE score. The tokenized generated captions are compared against the original captions for both the Neural as well as the CLIP Prefix model. 

In the second experiment, the stories generated are evaluated. 'Perplexity' is used to evaluate how well the model is able to predict/generate the next word.  The GPT-2 model is fine-tuned to generate coherent stories on Writing Prompts dataset. Further, multiple experiments were performed to analyse how coherent the stories are, when generated using a fine-tuned GPT-2 model as compared to without fine-tuning. Refer the two notebooks mentioned above.

The main experiment was to evaluate the entire pipeline for coherence and relevancy of the stories generated from the pair of three captions. An initial experiment was done for the baseline end-to-end pipeline, where the three captions (generated by the image captioning model) by the first model were used as input prompts to generate stories from the GPT-2 model. These results were evaluated manually for this experiment along with the separate evaluations. Then further the best model was chosen for the Image Captioning task and was integrated with the final, more-context aware Story Generation model to generate results and evaluated based on Automatic, newly created metrics. 

Hence, for the final system, the CLIP Prefix model was integrated with the GPT-2 model as they seemed to give the best results individually. For the final Story Generation model, changes were made to the initial GPT-2 model to improve the generation by introducing another GPT-2 model at the beginning to provide more context to the story model related to the images. This final model was then integrated with the Image Captioning model to generate coherent stories from a sequence of random images. A pre-trained GPT-2 model and tokenizer are used initially to generate a short story prompt in continuation to the given input of pair of three captions from the captioning model. The output along with the caption and the prompt are concatenated together, which is then passed to the fine-tuned GPT-2 model. This fine-tuned model then generates the story for each pair of three captions.For this experiment and the final model implemented, the corresponding code is present in the notebook [Image-to-Story System](CLIP Prefix_Zero-shot_Fine-tuned GPT-2_Img-to-Story.ipynb)


## Evaluation
The evaluations are done along with the code at the end in the notebooks. In addition to the Automatic evaluations done here, 

### Image Captioning Evaluation
The BLUE score is used to evaluate the performance of image captioning models. The CLIP Prefix model performs better than the Neural model as it has a BLUE score of 0.275 vs a BLUE score of 0.20 for the Neural model.

### Story Generation Evaluation
Perplexity score and human evaluation are used to assess the coherence and context-awareness of the stories. Based on these scores, Fine-tuned GPT-2 model performed really well for the Story Generation task.

### Image-to-Story System Evaluation
Reference-free metrics like TTR (lexical diversity), Fluency, Repetition, and Complexity are used. A new metric, WorLap, is created to evaluate the coherence between text generated and the themes presented in the images as well. The scores for the final system are presented in the [Image-to-Story System](CLIP Prefix_Zero-shot_Fine-tuned GPT-2_Img-to-Story.ipynb) notebook.
