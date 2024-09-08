<?php

namespace LeoPersan\ML;

class Layer
{
    /** @var Neuron[] */
    private array $neurons = [];  // Armazena os neurônios da camada
    /** @var float[] */
    private array $inputs = [];   // Armazena os inputs recebidos pela camada
    /** @var float[] */
    private array $outputs = [];  // Armazena os outputs gerados pela camada
    private ActivationFunction $activationFunction; // Função de ativação usada na camada

    // Construtor define a quantidade de neurônios e a função de ativação
    public function __construct($numInputs, $numNeurons, ActivationFunction $activationFunction)
    {
        $this->activationFunction = $activationFunction;

        // Cria os neurônios e inicializa cada um
        for ($i = 0; $i < $numNeurons; $i ++) {
            $this->neurons[] = new Neuron($numInputs);
        }
    }

    // Propaga os dados de entrada para frente e retorna os outputs
    public function forward(array $inputs)
    {
        $this->inputs  = $inputs;
        $this->outputs = [];

        // Calcula o output de cada neurônio
        foreach ($this->neurons as $neuron) {
            $output          = $neuron->activate($inputs, $this->activationFunction);
            $this->outputs[] = $output;
        }

        return $this->outputs;
    }

    // Executa a retropropagação, ajustando os pesos dos neurônios
    public function backward(array $error, $learningRate)
    {
        $newError = [];

        // Retropropaga o erro para cada neurônio e ajusta os pesos
        foreach ($this->neurons as $i => $neuron) {
            $newError[$i] = $neuron->adjustWeights($this->inputs, $error[$i], $learningRate, $this->activationFunction);
        }

        foreach ($newError as $row) {
            foreach ($row as $j => $value) {
                $newError[$j][] = $value;
            }
        }

        foreach ($newError as $i => $row) {
            $newError[$i] = array_sum($row) / count($row);
        }

        return $newError;
    }

    /**
     * @return Neuron[]
     */
    public function getNeurons(): array
    {
        return $this->neurons;
    }
}
