# Product Ratings

This application contains a simple embedding-based model that predicts ratings given a user and a product.

## The Model Structure

 (Input:           1:2:2 ) # user_id, product_id
 (split:           1:1:1,                   1:1:1 )
 (user_embed:      1:1:5 ), (product_embed:    1:1:5 )
 (concat:          2:1:5 ),
 (Flatten:         1:1:10),
 (FullyConnected:  1:1:128),
 (Relu:            1:1:128),
 (FullyConnected:  1:1:32),
 (Relu:            1:1:32),
 (FullyConnected:  1:1:1 ) # predicted ratings

## Input Data Format

Input data should be formatted as below

```
(int) (int) (float)
(int) (int) (float)
(int) (int) (float)
(int) (int) (float)
```
Each represents user_id, product_id and ratings, respectively.

Example, `product_ratings.txt` contains fake ratings on 5 userid X 5 products

## How to run a train epoch

Once you compile, with `meson`, you can run with `meson test nntrainer_product_ratings`.
Please file an issue if you have a problem running the example.

```bash
$ meson test nntrainer_product_ratings -v -c ${build_dir}
```

Expected output is as below...

```bash
#1/100 - Training Loss: 0.692463 >> [ Accuracy: 100% - Validation Loss : 0.690833 ]
...
#100/100 - Training Loss: 0.535373 >> [ Accuracy: 100% - Validation Loss : 0.53367 ]
```
