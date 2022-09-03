# RadioSignals_PyTorch
 РАСПОЗНАВАНИЕ СПЕКТРОВ ИМПУЛЬСНЫХ РАДИОЛОКАЦИОННЫХ СИГНАЛОВ 
С ПОМОЩЬЮ НЕЙРОННОЙ СЕТИ

Сигналы радиолокационных станций (РЛС) имеют разную структуру и, соответственно, разные спектры [1, 2]. Возьмем для примера три широко применяемых в радиолокации типа импульсных сигналов [2]: 

![image](https://user-images.githubusercontent.com/111675267/188273165-fc1e9c58-58b2-41a1-ad0a-a08258ca5900.png)
Рис. 1. Простой немодулированный радиоимпульс

![image](https://user-images.githubusercontent.com/111675267/188273191-70f84177-6910-4de2-b7e8-649c7fc8bbc9.png)
Рис. 2. Импульс с линейной частотной модуляцией (ЛЧМ)

![image](https://user-images.githubusercontent.com/111675267/188273217-30566b21-6b00-48dd-84fe-b7e6eb82b94f.png)
Рис. 3. Импульс с фазовой кодовой манипуляцией (ФКМ)

Спектры (модули амплитудных спектральных функций) перечисленных сигналов, как известно, имеют следующий вид [2]:

![image](https://user-images.githubusercontent.com/111675267/188273268-8814f36e-e7ce-46d2-a4ae-2bac8d43fd90.png)
Рис. 4. Спектр простого немодулированного радиоимпульса

![image](https://user-images.githubusercontent.com/111675267/188273288-f5cb3f21-efaf-4a16-a051-cd72cc23b8b4.png)
Рис. 5. Спектр ЛЧМ-сигнала

![image](https://user-images.githubusercontent.com/111675267/188273304-6b75cbb8-4e28-430f-b0ca-68fb1537a220.png)
Рис. 6. Спектр ФКМ-сигнала

Решим задачу распознавания данных спектров. Для этого используем сверточную нейронную сеть (CNN). Как известно [3, 4, 5], сети такого типа наилучшим образом подходят для распознавания изображений.  
Моделирование будем проводить с использованием фреймворка PyTorch [3].
Спроектируем простую VGG-образную CNN [3, 4] по следующей схеме (рис.7):

![image](https://user-images.githubusercontent.com/111675267/188273370-fa9ebb87-0556-4173-9ec4-cd31f8e0499d.png)
Рис. 7. Сверточная VGG-образная нейронная сеть

Каждый блок сети состоит последовательно из сверточного слоя, слоев активации, пакетной нормализации, а также подвыборочного и прореживающего слоев. Завершают сеть выравнивающий, полносвязный и выходной слои. 
Первый блок содержит 32 элемента свертки с ядром 3х3, второй – 64, последний– 128. Используемые здесь функции активации - “relu”, подвыборочные слои – “maxpooling (2х2)”, прореживание – 25%. После выравнивающего слоя использован 512-элементный полносвязный слой с последующим 50%-м прореживанием. На выходе имеем трехэлементный полносвязный слой с последующей активацией “softmax”.

С помощью сверточной нейронной сети мы получили точные результаты. Все спектры распознаются правильно, в соответствии со своими классами.
Таким образом, можно сделать следующий вывод: для распознавания спектров радиолокационных сигналов целесообразно использовать достаточно несложную сверточную нейронную сеть. 

Структура проекта (папки/файлы):
![structure](https://user-images.githubusercontent.com/111675267/188273770-03a8da82-494f-4ae7-b965-c727dcf51a0e.jpg)
