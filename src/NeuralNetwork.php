<?php

namespace LeoPersan\ML;

class NeuralNetwork
{
    /** @var Layer[]  */
    private array $layers = []; // Armazena as camadas da rede
    private float $learningRate; // Taxa de aprendizado

    private LossFunction $lossFunction; // Função de perda usada para treinamento

    // Construtor define a taxa de aprendizado
    public function __construct($learningRate = 0.01)
    {
        $this->learningRate = $learningRate;
    }

    // Adiciona uma camada à rede
    public function addLayer(Layer $layer)
    {
        $this->layers[] = $layer;
    }

    // Propaga os dados de entrada para a frente através das camadas
    public function forward(array $input)
    {
        $output = $input;

        // Passa os dados por cada camada
        foreach ($this->layers as $layer) {
            $output = $layer->forward($output);
        }

        return $output;
    }

    // Executa o processo de treinamento
    public function train(array $input, array $target, $epochs = 1000, $lossFunction = null, $batchSize = 1)
    {
        $this->lossFunction = $lossFunction ?? new LossFunction();
        [$input, $target] = $this->shuffle($input, $target);
        $batches = array_chunk($input, $batchSize);
        $targetBatches = array_chunk($target, $batchSize);
        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            // echo "Epoch: $epoch" . PHP_EOL;
            foreach ($batches as $bKey => $batch) {
                $output = [];
                foreach ($batch as $i => $row) {
                    $output[$i] = $this->forward($row);
                    // echo "Input: " . implode(', ', $row) . PHP_EOL;
                    // echo "Output: " . implode(', ', $target[$i]) . PHP_EOL;
                    // echo "Predict: " . implode(', ', $output[$i]) . PHP_EOL;
                    // echo PHP_EOL;
                }
                $this->backpropagate($output, $targetBatches[$bKey]);
            }
        }
    }

    // Realiza a retropropagação para ajustar os pesos
    private function backpropagate(array $output, array $target)
    {
        $error = $this->lossFunction->calculate($output, $target);
        // echo "Error: " . implode(', ', $error) . PHP_EOL;

        // Processa as camadas ao contrário para retropropagação
        for ($i = count($this->layers) - 1; $i >= 0; $i--) {
            $error = $this->layers[$i]->backward($error, $this->learningRate);
        }

        // $this->printWeights();
    }

    // Avalia a rede com um conjunto de dados
    public function predict(array $input)
    {
        return $this->forward($input);
    }

    // Imprime os pesos da rede
    public function printWeights()
    {
        foreach ($this->layers as $i => $layer) {
            echo "Layer $i" . PHP_EOL;
            foreach ($layer->getNeurons() as $j => $neuron) {
                echo "Neuron $j: " . implode(', ', $neuron->getWeights()) . PHP_EOL;
            }
        }
        echo PHP_EOL;
    }

    public function shuffle(array $input, array $target): array
    {
        $order = range(1, count($input));
        shuffle($order);
        array_multisort($order, $input, $target);

        return [$input, $target];
    }

    public function saveModel(string $string): int|false
    {
        $model = [];
        foreach ($this->layers as $i => $layer) {
            $model[$i] = [];
            foreach ($layer->getNeurons() as $j => $neuron) {
                $model[$i][$j] = $neuron->getWeights();
            }
        }

        return file_put_contents($string, json_encode($model));
    }
}
