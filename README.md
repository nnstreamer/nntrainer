# Transfer-Learning

I made some toy examples which is similar with Apple's Sticker.
The Mobile ssd V2 tensor flow lite model is used for the feature extractor and Nearest Neighbor is used for the classifier. All the training and testing is done on the Galaxy S8.

![image](https://github.sec.samsung.net/storage/user/19415/files/08b09a80-ef29-11e9-8303-475fd75f4b83)

Happy(^^), sad(TT), soso(ㅡㅡ) classes are used and prepare 5 images for the training and two images for the test set each as below.

![image](https://github.sec.samsung.net/storage/user/19415/files/a73cfb80-ef29-11e9-9ae9-0d6531538eaf)

After remove the fully connected layer of mobile ssd v2, 128 features are extracted. The features from first training set data is below.

![image](https://github.sec.samsung.net/storage/user/19415/files/0997fb00-ef2e-11e9-90a3-51c27bf4013f)


Simple euclidean distance is calculated and the result is quite good. All the test set is collected.

![image](https://github.sec.samsung.net/storage/user/19415/files/87103b00-ef2f-11e9-9c1a-83da0faafb63)

Due to the simplicity of this toy example, all the test results are collect.

I made two more random pictures which little bit differ from right image. As you can see, it is little bit hard to tell which class it is. First image could be classified as "happy" but the red zone is across with sad and the variance is quite small. Second image is more confused. Cause the smallest distance is all over the classes.
May be should be define the threshold which I didn't.^^;;

![image](https://github.sec.samsung.net/storage/user/19415/files/33552000-ef36-11e9-88f6-ea6a35ccdf6b)
