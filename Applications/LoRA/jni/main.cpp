#include <cifar_dataloader.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

const int number_of_db = 10;
const int batch_size = 10;
const int epochs = 2;
const float learning_rate = 0.001;

const int input_dim = 5;
const int unit_dim1 = 10;
const int unit_dim2 = 5;
const int rank = 3;
const float scaling = 2.0 / rank;
const int alpha = 2;

/** create test model */
std::unique_ptr<ml::train::Model> create_model() {
  std::unique_ptr<ml::train::Model> model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});

  model->addLayer(ml::train::createLayer(
    "input", {"name=input", "input_shape=1:1:" + std::to_string(input_dim)}));

  model->addLayer(ml::train::createLayer(
    "fully_connected",
    {"input_layers=input",
     "name=fc0",
     "unit=" + std::to_string(unit_dim1), "lora_rank=" + std::to_string(rank),
     "lora_alpha=" + std::to_string(alpha), "disable_bias=true"}));
  model->addLayer(ml::train::createLayer(
    "fully_connected",
    {"input_layers=fc0",
     "name=fc1",
     "unit=" + std::to_string(unit_dim2), "lora_rank=" + std::to_string(rank),
     "lora_alpha=" + std::to_string(alpha), "disable_bias=true"}));

  return model;
}

/** test dataset input */
std::unique_ptr<nntrainer::util::DataLoader> getRandomDataGenerator() {
  std::unique_ptr<nntrainer::util::DataLoader> random_db(
    new nntrainer::util::RandomDataLoader(
      {{batch_size, 1, 1, input_dim}}, {{batch_size, 1, 1, 1}}, number_of_db));
  return random_db;
}

/** test dataset generator */
std::unique_ptr<nntrainer::util::DataLoader> getTestDataGenerator() {
  std::unique_ptr<nntrainer::util::DataLoader> test_db(
    new nntrainer::util::OnesTestDataLoader(
      {{batch_size, 1, 1, input_dim}}, {{batch_size, 1, 1, 10}}, number_of_db));
  return test_db;
}

/** dataset callback */
int dataset_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

int main(int argc, char *argv[]) {

  /** a single fc layer with LoRA */
  auto model = create_model();

  model->setProperty({"batch_size=" + std::to_string(batch_size),
                      "epochs=" + std::to_string(epochs),
                      "save_path=my_app.bin"});

  auto optimizer = ml::train::createOptimizer(
    "SGD", {"learning_rate=" + std::to_string(learning_rate)});
  model->setOptimizer(std::move(optimizer));

  int status = model->compile();
  status = model->initialize();

  // auto random_generator = getRandomDataGenerator();
  auto testdata_generator = getTestDataGenerator();
  auto train_dataset = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, dataset_cb, testdata_generator.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(train_dataset));
  model->train();

  return status;
}
