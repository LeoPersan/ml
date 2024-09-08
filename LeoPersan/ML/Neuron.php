<?php

namespace LeoPersan\ML;
class Neuron
{
    private $weights = [];  // Pesos associados às entradas do neurônio
    private $bias;          // O bias do neurônio
    private $output;        // Saída atual do neurônio

    // Inicializa os pesos e bias aleatoriamente
    public function __construct($numInputs)
    {
        // Inicializa os pesos aleatoriamente entre -1 e 1
        for ($i = 0; $i < $numInputs; $i++) {
            $this->weights[] = rand(-1000, 1000) / 1000;
        }

        // Inicializa o bias aleatoriamente
        $this->bias = rand(-1000, 1000) / 1000;
    }

    // Realiza a ativação do neurônio com base nas entradas e na função de ativação
    public function activate(array $inputs, ActivationFunction $activationFunction)
    {
        // Calcula a soma ponderada das entradas e pesos
        $sum = 0;
        foreach ($inputs as $i => $input) {
            $sum += $input * $this->weights[$i];
        }

        // Adiciona o bias à soma ponderada
        $sum += $this->bias;

        // Aplica a função de ativação
        $this->output = $activationFunction->apply($sum);

        return $this->output;
    }

    // Ajusta os pesos com base no erro, a taxa de aprendizado e a função de ativação
    public function adjustWeights(array $inputs, float $error, $learningRate, ActivationFunction $activationFunction)
    {
        // Derivada da função de ativação para a retropropagação
        $delta = $error * $activationFunction->derivative($this->output);

        // Calcula o novo erro para a camada anterior
        $newError = [];
        foreach ($this->weights as $i => $weight) {
            $newError[] = $delta * $weight;
        }

        // Ajusta os pesos e o bias
        foreach ($this->weights as $i => $weight) {
            $this->weights[$i] -= $learningRate * $delta * $inputs[$i];
        }

        // Ajusta o bias
        $this->bias -= $learningRate * $delta;

        return $newError;
    }

    /**
     * @return float[]
     */
    public function getWeights(): array
    {
        return $this->weights;
    }

    public function setWeights(array $weights)
    {
        $this->weights = $weights;
    }
}
