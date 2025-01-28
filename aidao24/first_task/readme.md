пояснения:

папка parc_mapping - код для маппинга парцелляций атласа Brainnetome к парцелляциям Schaefer200 -> на выходе получаем данные в 1 атласе
затем применяем наш алгоритм для получения кластеров по каждому человеку, ответ записывается в файл submission.csv

для получения ответов в 1 задаче:

```bash
python3 -m venv .venv
.venv/scripts/activate
pip install -r requirements.txt
python3 ./parc_mapping/map_data.py
python3 ./first_task.py
```