# 🤖 Fizzbuzz neural network

FizzBuzzNN es una solución al clásico problema de FizzBuzz utilizando una red neuronal. En lugar de implementar la lógica tradicional con condicionales, un modelo fue entrenado para que aprenda a predecir la salida correcta de FizzBuzz en función de un número de entrada.

## 🔢 Problema FizzBuzz

FizzBuzz es un problema común en programación donde, dados los números del 1 al 100, se imprime:
* "Fizz" si el número es divisible por 3.
* "Buzz" si el número es divisible por 5.
* "FizzBuzz" si el número es divisible por 15.
*  El número en otros casos.

## 🧠 Solución

En lugar de codificar reglas explícitas, se entrenó una red neuronal para que aprenda la lógica de tras el problema "FizzBuzz". El enfoque es el siguiente:
* Representar los números de entrada en una forma adecuada para la red neuronal.
* Etiquetar los datos con las salidas esperadas.
* Entrenar una red neuronal para aprender la relación entre entrada y salida.
* Evaluar las predicciones acertadas y probar su desempeño en nuevos datos.

## ⚙️ Implementación
1) Codificación de la entrada: Se convierte cada número en una representación binaria.
2) Red Neuronal: Una red con distintas capas aprende a clasificar los números en una de cuatro categorías: Fizz, Buzz, FizzBuzz, o el número.
3) Entrenamiento: Se utilizan datos del 1 al 100 para entrenar la red.
4) Predicción: Se evalúa el modelo en datos no vistos para verificar su rendimiento.

## 📂 Estructura del Proyecto
```
📁 fizz-buzz-NN
├── 📂 fizz_buzz 
    ├── __init__.py             
    ├── config.py
    ├── model.py
    ├── test_utils.py 
    ├── utils.py   
├── 📂 src                         
        ├── predict.py             
        ├── train.py   
├── .gitignore       
├── .gitattributes
├── README.md  
├── requirements.txt 
├── setup.py
```    

## 🚀 Instalación

Clona el repositorio:
```
git clone https://github.com/jeremias-mena/fizz-buzz-NN.git
```
```
cd fizz-buzz-NN
```

Instala las dependencias para poder utilizar este repositorio:
```
pip install -r requirements.txt
```

## 🏃 Uso
Entrenar el modelo
```
python train.py
```

Realizar predicciones
```
python predict.py
```

## 📫 Contacto
Si tienes preguntas o sugerencias, no dudes en abrir un issue o contactarme en menajeremias08@gmail.com.