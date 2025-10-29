subdir('jni')
subdir('datagen/cifar')
subdir('npy_reader')

#for now support ubuntu, andoird
if (get_option('platform') != 'windows') and (get_option('platform') != 'tizen')
    subdir('llm_tokenizer')
endif
