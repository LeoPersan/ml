<?php

namespace LeoPersan\ML;
class ActivationFunction
{
    public function apply($sum)
    {
        return 1 / (1 + exp(-$sum));
    }

    public function derivative($output)
    {
        return $output * (1 - $output);
    }
}
