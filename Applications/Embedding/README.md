# Embedding

This application contains embedding layer example with two example model.
One with single input, another assuming there are two inputs(using split layer)

## Example Model Structure

There are two structure ready for the example

1. Linear Embedding structure
 (Input:           10:1:1:4 )
 (Embedding:       10:1:4:8 )
 (Flatten:         10:1:1:32)
 (FullyConnected:  10:1:1:1 )

2. Splitted data and seperate embedding structure
 (Input:           10:1:2:2 )
 (split:           10:1:1:1,                   10:1:1:1 )
 (Embedding1:      10:1:1:8 ), (Embedding2:    10:1:1:8 )
 (concat:          10:2:1:8 ),
 (Flatten:         10:1:1:16),
 (FullyConnected:  10:1:1:1 )

## How to run a train epoch

Once you compile, with `meson`, please file an issue if you have a problem running the example.

```bash
export ${res}
$ cd ${build_dir}
$ meson test app_embedding -v #this is for the first model structure
$ meson test app_embedding_split -v #this is for the second model structure
```

Expected output is as below...

```bash
#1/100 - Training Loss: 0.692463 >> [ Accuracy: 100% - Validation Loss : 0.690833 ]
...
#100/100 - Training Loss: 0.535373 >> [ Accuracy: 100% - Validation Loss : 0.53367 ]
```
