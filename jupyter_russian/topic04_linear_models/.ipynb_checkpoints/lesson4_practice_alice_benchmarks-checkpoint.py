{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<center>\n",
    "<img src=\"../../img/ods_stickers.jpg\">\n",
    "## Открытый курс по машинному обучению. Сессия № 2\n",
    "</center>\n",
    "Автор материала: Юрий Исаков и Юрий Кашницкий. Материал распространяется на условиях лицензии [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Можно использовать в любых целях (редактировать, поправлять и брать за основу), кроме коммерческих, но с обязательным упоминанием автора материала."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# <center>Тема 4. Линейные модели классификации и регрессии\n",
    "## <center>  Практика. Идентификация пользователя с помощью логистической регрессии\n",
    "\n",
    "Тут мы воспроизведем парочку бенчмарков нашего соревнования и вдохновимся побить третий бенчмарк, а также остальных участников. Веб-формы для отправки ответов тут не будет, ориентир – [leaderboard](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/leaderboard) соревнования."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from scipy.sparse import csr_matrix, hstack\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import roc_auc_score\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from tqdm import tqdm_notebook\n",
    "\n",
    "%matplotlib inline\n",
    "import seaborn as sns\n",
    "from matplotlib import pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Загрузка и преобразование данных\n",
    "Зарегистрируйтесь на [Kaggle](www.kaggle.com), если вы не сделали этого раньше, зайдите на [страницу](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) соревнования и скачайте данные. Первым делом загрузим обучающую и тестовую выборки и посмотрим на данные."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>site1</th>\n",
       "      <th>time1</th>\n",
       "      <th>site2</th>\n",
       "      <th>time2</th>\n",
       "      <th>site3</th>\n",
       "      <th>time3</th>\n",
       "      <th>site4</th>\n",
       "      <th>time4</th>\n",
       "      <th>site5</th>\n",
       "      <th>time5</th>\n",
       "      <th>...</th>\n",
       "      <th>time6</th>\n",
       "      <th>site7</th>\n",
       "      <th>time7</th>\n",
       "      <th>site8</th>\n",
       "      <th>time8</th>\n",
       "      <th>site9</th>\n",
       "      <th>time9</th>\n",
       "      <th>site10</th>\n",
       "      <th>time10</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>session_id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>21669</th>\n",
       "      <td>56</td>\n",
       "      <td>2013-01-12 08:05:57</td>\n",
       "      <td>55.0</td>\n",
       "      <td>2013-01-12 08:05:57</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>...</td>\n",
       "      <td>NaT</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>54843</th>\n",
       "      <td>56</td>\n",
       "      <td>2013-01-12 08:37:23</td>\n",
       "      <td>55.0</td>\n",
       "      <td>2013-01-12 08:37:23</td>\n",
       "      <td>56.0</td>\n",
       "      <td>2013-01-12 09:07:07</td>\n",
       "      <td>55.0</td>\n",
       "      <td>2013-01-12 09:07:09</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>...</td>\n",
       "      <td>NaT</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaT</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>77292</th>\n",
       "      <td>946</td>\n",
       "      <td>2013-01-12 08:50:13</td>\n",
       "      <td>946.0</td>\n",
       "      <td>2013-01-12 08:50:14</td>\n",
       "      <td>951.0</td>\n",
       "      <td>2013-01-12 08:50:15</td>\n",
       "      <td>946.0</td>\n",
       "      <td>2013-01-12 08:50:15</td>\n",
       "      <td>946.0</td>\n",
       "      <td>2013-01-12 08:50:16</td>\n",
       "      <td>...</td>\n",
       "      <td>2013-01-12 08:50:16</td>\n",
       "      <td>948.0</td>\n",
       "      <td>2013-01-12 08:50:16</td>\n",
       "      <td>784.0</td>\n",
       "      <td>2013-01-12 08:50:16</td>\n",
       "      <td>949.0</td>\n",
       "      <td>2013-01-12 08:50:17</td>\n",
       "      <td>946.0</td>\n",
       "      <td>2013-01-12 08:50:17</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>114021</th>\n",
       "      <td>945</td>\n",
       "      <td>2013-01-12 08:50:17</td>\n",
       "      <td>948.0</td>\n",
       "      <td>2013-01-12 08:50:17</td>\n",
       "      <td>949.0</td>\n",
       "      <td>2013-01-12 08:50:18</td>\n",
       "      <td>948.0</td>\n",
       "      <td>2013-01-12 08:50:18</td>\n",
       "      <td>945.0</td>\n",
       "      <td>2013-01-12 08:50:18</td>\n",
       "      <td>...</td>\n",
       "      <td>2013-01-12 08:50:18</td>\n",
       "      <td>947.0</td>\n",
       "      <td>2013-01-12 08:50:19</td>\n",
       "      <td>945.0</td>\n",
       "      <td>2013-01-12 08:50:19</td>\n",
       "      <td>946.0</td>\n",
       "      <td>2013-01-12 08:50:19</td>\n",
       "      <td>946.0</td>\n",
       "      <td>2013-01-12 08:50:20</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146670</th>\n",
       "      <td>947</td>\n",
       "      <td>2013-01-12 08:50:20</td>\n",
       "      <td>950.0</td>\n",
       "      <td>2013-01-12 08:50:20</td>\n",
       "      <td>948.0</td>\n",
       "      <td>2013-01-12 08:50:20</td>\n",
       "      <td>947.0</td>\n",
       "      <td>2013-01-12 08:50:21</td>\n",
       "      <td>950.0</td>\n",
       "      <td>2013-01-12 08:50:21</td>\n",
       "      <td>...</td>\n",
       "      <td>2013-01-12 08:50:21</td>\n",
       "      <td>946.0</td>\n",
       "      <td>2013-01-12 08:50:21</td>\n",
       "      <td>951.0</td>\n",
       "      <td>2013-01-12 08:50:22</td>\n",
       "      <td>946.0</td>\n",
       "      <td>2013-01-12 08:50:22</td>\n",
       "      <td>947.0</td>\n",
       "      <td>2013-01-12 08:50:22</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "            site1               time1  site2               time2  site3  \\\n",
       "session_id                                                                \n",
       "21669          56 2013-01-12 08:05:57   55.0 2013-01-12 08:05:57    NaN   \n",
       "54843          56 2013-01-12 08:37:23   55.0 2013-01-12 08:37:23   56.0   \n",
       "77292         946 2013-01-12 08:50:13  946.0 2013-01-12 08:50:14  951.0   \n",
       "114021        945 2013-01-12 08:50:17  948.0 2013-01-12 08:50:17  949.0   \n",
       "146670        947 2013-01-12 08:50:20  950.0 2013-01-12 08:50:20  948.0   \n",
       "\n",
       "                         time3  site4               time4  site5  \\\n",
       "session_id                                                         \n",
       "21669                      NaT    NaN                 NaT    NaN   \n",
       "54843      2013-01-12 09:07:07   55.0 2013-01-12 09:07:09    NaN   \n",
       "77292      2013-01-12 08:50:15  946.0 2013-01-12 08:50:15  946.0   \n",
       "114021     2013-01-12 08:50:18  948.0 2013-01-12 08:50:18  945.0   \n",
       "146670     2013-01-12 08:50:20  947.0 2013-01-12 08:50:21  950.0   \n",
       "\n",
       "                         time5  ...               time6  site7  \\\n",
       "session_id                      ...                              \n",
       "21669                      NaT  ...                 NaT    NaN   \n",
       "54843                      NaT  ...                 NaT    NaN   \n",
       "77292      2013-01-12 08:50:16  ... 2013-01-12 08:50:16  948.0   \n",
       "114021     2013-01-12 08:50:18  ... 2013-01-12 08:50:18  947.0   \n",
       "146670     2013-01-12 08:50:21  ... 2013-01-12 08:50:21  946.0   \n",
       "\n",
       "                         time7  site8               time8  site9  \\\n",
       "session_id                                                         \n",
       "21669                      NaT    NaN                 NaT    NaN   \n",
       "54843                      NaT    NaN                 NaT    NaN   \n",
       "77292      2013-01-12 08:50:16  784.0 2013-01-12 08:50:16  949.0   \n",
       "114021     2013-01-12 08:50:19  945.0 2013-01-12 08:50:19  946.0   \n",
       "146670     2013-01-12 08:50:21  951.0 2013-01-12 08:50:22  946.0   \n",
       "\n",
       "                         time9 site10              time10 target  \n",
       "session_id                                                        \n",
       "21669                      NaT    NaN                 NaT      0  \n",
       "54843                      NaT    NaN                 NaT      0  \n",
       "77292      2013-01-12 08:50:17  946.0 2013-01-12 08:50:17      0  \n",
       "114021     2013-01-12 08:50:19  946.0 2013-01-12 08:50:20      0  \n",
       "146670     2013-01-12 08:50:22  947.0 2013-01-12 08:50:22      0  \n",
       "\n",
       "[5 rows x 21 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# загрузим обучающую и тестовую выборки\n",
    "train_df = pd.read_csv(\"../../data/train_sessions.csv\", index_col=\"session_id\")\n",
    "test_df = pd.read_csv(\"../../data/test_sessions.csv\", index_col=\"session_id\")\n",
    "\n",
    "# приведем колонки time1, ..., time10 к временному формату\n",
    "times = [\"time%s\" % i for i in range(1, 11)]\n",
    "train_df[times] = train_df[times].apply(pd.to_datetime)\n",
    "test_df[times] = test_df[times].apply(pd.to_datetime)\n",
    "\n",
    "# отсортируем данные по времени\n",
    "train_df = train_df.sort_values(by=\"time1\")\n",
    "\n",
    "# посмотрим на заголовок обучающей выборки\n",
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "В обучающей выборке содержатся следующие признаки:\n",
    "    - site1 – индекс первого посещенного сайта в сессии\n",
    "    - time1 – время посещения первого сайта в сессии\n",
    "    - ...\n",
    "    - site10 – индекс 10-го посещенного сайта в сессии\n",
    "    - time10 – время посещения 10-го сайта в сессии\n",
    "    - target – целевая переменная, 1 для сессий Элис, 0 для сессий других пользователей\n",
    "    \n",
    "Сессии пользователей выделены таким образом, что они не могут быть длиннее получаса или 10 сайтов. То есть сессия считается оконченной либо когда пользователь посетил 10 сайтов подряд либо когда сессия заняла по времени более 30 минут.\n",
    "\n",
    "В таблице встречаются пропущенные значения, это значит, что сессия состоит менее, чем из 10 сайтов. Заменим пропущенные значения нулями и приведем признаки к целому типу. Также загрузим словарь сайтов и посмотрим, как он выглядит:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "всего сайтов: 48371\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>site</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>25075</th>\n",
       "      <td>www.abmecatronique.com</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13997</th>\n",
       "      <td>groups.live.com</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>42436</th>\n",
       "      <td>majeureliguefootball.wordpress.com</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30911</th>\n",
       "      <td>cdt46.media.tourinsoft.eu</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8104</th>\n",
       "      <td>www.hdwallpapers.eu</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                     site\n",
       "25075              www.abmecatronique.com\n",
       "13997                     groups.live.com\n",
       "42436  majeureliguefootball.wordpress.com\n",
       "30911           cdt46.media.tourinsoft.eu\n",
       "8104                  www.hdwallpapers.eu"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# приведем колонки site1, ..., site10 к целочисленному формату и заменим пропуски нулями\n",
    "sites = [\"site%s\" % i for i in range(1, 11)]\n",
    "train_df[sites] = train_df[sites].fillna(0).astype(\"int\")\n",
    "test_df[sites] = test_df[sites].fillna(0).astype(\"int\")\n",
    "\n",
    "# загрузим словарик сайтов\n",
    "with open(r\"../../data/site_dic.pkl\", \"rb\") as input_file:\n",
    "    site_dict = pickle.load(input_file)\n",
    "\n",
    "# датафрейм словарика сайтов\n",
    "sites_dict_df = pd.DataFrame(\n",
    "    list(site_dict.keys()), index=list(site_dict.values()), columns=[\"site\"]\n",
    ")\n",
    "print(u\"всего сайтов:\", sites_dict_df.shape[0])\n",
    "sites_dict_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Выделим целевую переменную и объединим выборки, чтобы вместе привести их к разреженному формату."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# наша целевая переменная\n",
    "y_train = train_df[\"target\"]\n",
    "\n",
    "# объединенная таблица исходных данных\n",
    "full_df = pd.concat([train_df.drop(\"target\", axis=1), test_df])\n",
    "\n",
    "# индекс, по которому будем отделять обучающую выборку от тестовой\n",
    "idx_split = train_df.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(336358, 20)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(253561,)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Для самой первой модели будем использовать только посещенные сайты в сессии (но не будем обращать внимание на временные признаки). За таким выбором данных для модели стоит такая идея:  *у Элис есть свои излюбленные сайты, и чем чаще вы видим эти сайты в сессии, тем выше вероятность, что это сессия Элис и наоборот.*\n",
    "\n",
    "Подготовим данные, из всей таблицы выберем только признаки `site1, site2, ... , site10`. Напомним, что пропущенные значения заменены нулем. Вот как выглядят первые строки таблицы:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>site1</th>\n",
       "      <th>site2</th>\n",
       "      <th>site3</th>\n",
       "      <th>site4</th>\n",
       "      <th>site5</th>\n",
       "      <th>site6</th>\n",
       "      <th>site7</th>\n",
       "      <th>site8</th>\n",
       "      <th>site9</th>\n",
       "      <th>site10</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>session_id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>21669</th>\n",
       "      <td>56</td>\n",
       "      <td>55</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>54843</th>\n",
       "      <td>56</td>\n",
       "      <td>55</td>\n",
       "      <td>56</td>\n",
       "      <td>55</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>77292</th>\n",
       "      <td>946</td>\n",
       "      <td>946</td>\n",
       "      <td>951</td>\n",
       "      <td>946</td>\n",
       "      <td>946</td>\n",
       "      <td>945</td>\n",
       "      <td>948</td>\n",
       "      <td>784</td>\n",
       "      <td>949</td>\n",
       "      <td>946</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>114021</th>\n",
       "      <td>945</td>\n",
       "      <td>948</td>\n",
       "      <td>949</td>\n",
       "      <td>948</td>\n",
       "      <td>945</td>\n",
       "      <td>946</td>\n",
       "      <td>947</td>\n",
       "      <td>945</td>\n",
       "      <td>946</td>\n",
       "      <td>946</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146670</th>\n",
       "      <td>947</td>\n",
       "      <td>950</td>\n",
       "      <td>948</td>\n",
       "      <td>947</td>\n",
       "      <td>950</td>\n",
       "      <td>952</td>\n",
       "      <td>946</td>\n",
       "      <td>951</td>\n",
       "      <td>946</td>\n",
       "      <td>947</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            site1  site2  site3  site4  site5  site6  site7  site8  site9  \\\n",
       "session_id                                                                  \n",
       "21669          56     55      0      0      0      0      0      0      0   \n",
       "54843          56     55     56     55      0      0      0      0      0   \n",
       "77292         946    946    951    946    946    945    948    784    949   \n",
       "114021        945    948    949    948    945    946    947    945    946   \n",
       "146670        947    950    948    947    950    952    946    951    946   \n",
       "\n",
       "            site10  \n",
       "session_id          \n",
       "21669            0  \n",
       "54843            0  \n",
       "77292          946  \n",
       "114021         946  \n",
       "146670         947  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# табличка с индексами посещенных сайтов в сессии\n",
    "full_sites = full_df[sites]\n",
    "full_sites.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Сессии представляют собой последовательность индексов сайтов и данные в таком виде неудобны для линейных методов. В соответствии с нашей гипотезой (у Элис есть излюбленные сайты) надо преобразовать эту таблицу таким образом, чтобы каждому возможному сайту соответствовал свой отдельный признак (колонка), а его значение равнялось бы количеству посещений этого сайта в сессии. Это делается в две строчки:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy.sparse import csr_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[0;31mInit signature:\u001b[0m \u001b[0mcsr_matrix\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0marg1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mshape\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmaxprint\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
       "\u001b[0;31mDocstring:\u001b[0m     \n",
       "Compressed Sparse Row matrix.\n",
       "\n",
       "This can be instantiated in several ways:\n",
       "    csr_matrix(D)\n",
       "        where D is a 2-D ndarray\n",
       "\n",
       "    csr_matrix(S)\n",
       "        with another sparse array or matrix S (equivalent to S.tocsr())\n",
       "\n",
       "    csr_matrix((M, N), [dtype])\n",
       "        to construct an empty matrix with shape (M, N)\n",
       "        dtype is optional, defaulting to dtype='d'.\n",
       "\n",
       "    csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])\n",
       "        where ``data``, ``row_ind`` and ``col_ind`` satisfy the\n",
       "        relationship ``a[row_ind[k], col_ind[k]] = data[k]``.\n",
       "\n",
       "    csr_matrix((data, indices, indptr), [shape=(M, N)])\n",
       "        is the standard CSR representation where the column indices for\n",
       "        row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their\n",
       "        corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.\n",
       "        If the shape parameter is not supplied, the matrix dimensions\n",
       "        are inferred from the index arrays.\n",
       "\n",
       "Attributes\n",
       "----------\n",
       "dtype : dtype\n",
       "    Data type of the matrix\n",
       "shape : 2-tuple\n",
       "    Shape of the matrix\n",
       "ndim : int\n",
       "    Number of dimensions (this is always 2)\n",
       "nnz\n",
       "size\n",
       "data\n",
       "    CSR format data array of the matrix\n",
       "indices\n",
       "    CSR format index array of the matrix\n",
       "indptr\n",
       "    CSR format index pointer array of the matrix\n",
       "has_sorted_indices\n",
       "has_canonical_format\n",
       "T\n",
       "\n",
       "Notes\n",
       "-----\n",
       "\n",
       "Sparse matrices can be used in arithmetic operations: they support\n",
       "addition, subtraction, multiplication, division, and matrix power.\n",
       "\n",
       "Advantages of the CSR format\n",
       "  - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.\n",
       "  - efficient row slicing\n",
       "  - fast matrix vector products\n",
       "\n",
       "Disadvantages of the CSR format\n",
       "  - slow column slicing operations (consider CSC)\n",
       "  - changes to the sparsity structure are expensive (consider LIL or DOK)\n",
       "\n",
       "Canonical Format\n",
       "    - Within each row, indices are sorted by column.\n",
       "    - There are no duplicate entries.\n",
       "\n",
       "Examples\n",
       "--------\n",
       "\n",
       ">>> import numpy as np\n",
       ">>> from scipy.sparse import csr_matrix\n",
       ">>> csr_matrix((3, 4), dtype=np.int8).toarray()\n",
       "array([[0, 0, 0, 0],\n",
       "       [0, 0, 0, 0],\n",
       "       [0, 0, 0, 0]], dtype=int8)\n",
       "\n",
       ">>> row = np.array([0, 0, 1, 2, 2, 2])\n",
       ">>> col = np.array([0, 2, 2, 0, 1, 2])\n",
       ">>> data = np.array([1, 2, 3, 4, 5, 6])\n",
       ">>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()\n",
       "array([[1, 0, 2],\n",
       "       [0, 0, 3],\n",
       "       [4, 5, 6]])\n",
       "\n",
       ">>> indptr = np.array([0, 2, 3, 6])\n",
       ">>> indices = np.array([0, 2, 2, 0, 1, 2])\n",
       ">>> data = np.array([1, 2, 3, 4, 5, 6])\n",
       ">>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()\n",
       "array([[1, 0, 2],\n",
       "       [0, 0, 3],\n",
       "       [4, 5, 6]])\n",
       "\n",
       "Duplicate entries are summed together:\n",
       "\n",
       ">>> row = np.array([0, 1, 2, 0])\n",
       ">>> col = np.array([0, 1, 1, 0])\n",
       ">>> data = np.array([1, 2, 4, 8])\n",
       ">>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()\n",
       "array([[9, 0, 0],\n",
       "       [0, 2, 0],\n",
       "       [0, 4, 0]])\n",
       "\n",
       "As an example of how to construct a CSR matrix incrementally,\n",
       "the following snippet builds a term-document matrix from texts:\n",
       "\n",
       ">>> docs = [[\"hello\", \"world\", \"hello\"], [\"goodbye\", \"cruel\", \"world\"]]\n",
       ">>> indptr = [0]\n",
       ">>> indices = []\n",
       ">>> data = []\n",
       ">>> vocabulary = {}\n",
       ">>> for d in docs:\n",
       "...     for term in d:\n",
       "...         index = vocabulary.setdefault(term, len(vocabulary))\n",
       "...         indices.append(index)\n",
       "...         data.append(1)\n",
       "...     indptr.append(len(indices))\n",
       "...\n",
       ">>> csr_matrix((data, indices, indptr), dtype=int).toarray()\n",
       "array([[2, 1, 0, 0],\n",
       "       [0, 1, 1, 1]])\n",
       "\u001b[0;31mFile:\u001b[0m           ~/my_ml_education/venv/lib/python3.13/site-packages/scipy/sparse/_csr.py\n",
       "\u001b[0;31mType:\u001b[0m           type\n",
       "\u001b[0;31mSubclasses:\u001b[0m     "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "csr_matrix?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# последовательность с индексами\n",
    "sites_flatten = full_sites.values.flatten()\n",
    "\n",
    "# искомая матрица\n",
    "full_sites_sparse = csr_matrix(\n",
    "    (\n",
    "        [1] * sites_flatten.shape[0],\n",
    "        sites_flatten,\n",
    "        range(0, sites_flatten.shape[0] + 10, 10),\n",
    "    )\n",
    ")[:, 1:]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Еще один плюс использования разреженных матриц в том, что для них имеются специальные реализации как матричных операций, так и алгоритмов машинного обучения, что подчас позволяет ощутимо ускорить операции за счет особенностей структуры данных. Это касается и логистической регрессии. Вот теперь у нас все готово для построения нашей первой модели.\n",
    "\n",
    "### 2. Построение первой модели\n",
    "\n",
    "Итак, у нас есть алгоритм и данные для него, построим нашу первую модель, воспользовавшись релизацией [логистической регрессии](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) из пакета `sklearn` с параметрами по умолчанию. Первые 90% данных будем использовать для обучения (обучающая выборка отсортирована по времени), а оставшиеся 10% для проверки качества (validation). \n",
    "\n",
    "**Напишите простую функцию, которая будет возвращать качество модели на отложенной выборке, и обучите наш первый классификатор**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import roc_auc_score\n",
    "\n",
    "def get_auc_lr_valid(X, y, C=1.0, ratio=0.9, seed=17):\n",
    "    \"\"\"\n",
    "    X, y — выборка\n",
    "    ratio — в каком отношении поделить выборку (доля train)\n",
    "    C, seed — коэффициент регуляризации и random_state для логистической регрессии\n",
    "    \"\"\"\n",
    "    \n",
    "    # Разбиение на train и test\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=seed)\n",
    "    \n",
    "    # Обучение модели\n",
    "    lr_clf = LogisticRegression(C=C, random_state=seed)\n",
    "    lr_clf.fit(X_train, y_train)\n",
    "    \n",
    "    # Предсказание вероятностей\n",
    "    y_pred_proba = lr_clf.predict_proba(X_test)  # Берем вероятность класса 1\n",
    "    \n",
    "    # Вычисление AUC-ROC\n",
    "    # auc = roc_auc_score(y_test, y_pred_proba)\n",
    "    \n",
    "    # print(auc)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Посмотрите, какой получился ROC AUC на отложенной выборке.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "ename": "DTypePromotionError",
     "evalue": "The DType <class 'numpy.dtypes.DateTime64DType'> could not be promoted by <class 'numpy.dtypes.Int64DType'>. This means that no common DType exists for the given inputs. For example they cannot be stored in a single array unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mDTypePromotionError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[26], line 2\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# Ваш код здесь\u001b[39;00m\n\u001b[0;32m----> 2\u001b[0m \u001b[43mget_auc_lr_valid\u001b[49m\u001b[43m(\u001b[49m\u001b[43mtrain_df\u001b[49m\u001b[43m,\u001b[49m\u001b[43my_train\u001b[49m\u001b[43m)\u001b[49m\n",
      "Cell \u001b[0;32mIn[25], line 17\u001b[0m, in \u001b[0;36mget_auc_lr_valid\u001b[0;34m(X, y, C, ratio, seed)\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[38;5;66;03m# Обучение модели\u001b[39;00m\n\u001b[1;32m     16\u001b[0m lr_clf \u001b[38;5;241m=\u001b[39m LogisticRegression(C\u001b[38;5;241m=\u001b[39mC, random_state\u001b[38;5;241m=\u001b[39mseed)\n\u001b[0;32m---> 17\u001b[0m \u001b[43mlr_clf\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX_train\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43my_train\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     19\u001b[0m \u001b[38;5;66;03m# Предсказание вероятностей\u001b[39;00m\n\u001b[1;32m     20\u001b[0m y_pred_proba \u001b[38;5;241m=\u001b[39m lr_clf\u001b[38;5;241m.\u001b[39mpredict_proba(X_test)\n",
      "File \u001b[0;32m~/my_ml_education/venv/lib/python3.13/site-packages/sklearn/base.py:1389\u001b[0m, in \u001b[0;36m_fit_context.<locals>.decorator.<locals>.wrapper\u001b[0;34m(estimator, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1382\u001b[0m     estimator\u001b[38;5;241m.\u001b[39m_validate_params()\n\u001b[1;32m   1384\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m config_context(\n\u001b[1;32m   1385\u001b[0m     skip_parameter_validation\u001b[38;5;241m=\u001b[39m(\n\u001b[1;32m   1386\u001b[0m         prefer_skip_nested_validation \u001b[38;5;129;01mor\u001b[39;00m global_skip_validation\n\u001b[1;32m   1387\u001b[0m     )\n\u001b[1;32m   1388\u001b[0m ):\n\u001b[0;32m-> 1389\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfit_method\u001b[49m\u001b[43m(\u001b[49m\u001b[43mestimator\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m~/my_ml_education/venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:1222\u001b[0m, in \u001b[0;36mLogisticRegression.fit\u001b[0;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[1;32m   1219\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m   1220\u001b[0m     _dtype \u001b[38;5;241m=\u001b[39m [np\u001b[38;5;241m.\u001b[39mfloat64, np\u001b[38;5;241m.\u001b[39mfloat32]\n\u001b[0;32m-> 1222\u001b[0m X, y \u001b[38;5;241m=\u001b[39m \u001b[43mvalidate_data\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m   1223\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1224\u001b[0m \u001b[43m    \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1225\u001b[0m \u001b[43m    \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1226\u001b[0m \u001b[43m    \u001b[49m\u001b[43maccept_sparse\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mcsr\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1227\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdtype\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43m_dtype\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1228\u001b[0m \u001b[43m    \u001b[49m\u001b[43morder\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mC\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1229\u001b[0m \u001b[43m    \u001b[49m\u001b[43maccept_large_sparse\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43msolver\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;129;43;01mnot\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[38;5;129;43;01min\u001b[39;49;00m\u001b[43m \u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mliblinear\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43msag\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43msaga\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1230\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1231\u001b[0m check_classification_targets(y)\n\u001b[1;32m   1232\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mclasses_ \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39munique(y)\n",
      "File \u001b[0;32m~/my_ml_education/venv/lib/python3.13/site-packages/sklearn/utils/validation.py:2961\u001b[0m, in \u001b[0;36mvalidate_data\u001b[0;34m(_estimator, X, y, reset, validate_separately, skip_check_array, **check_params)\u001b[0m\n\u001b[1;32m   2959\u001b[0m         y \u001b[38;5;241m=\u001b[39m check_array(y, input_name\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124my\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mcheck_y_params)\n\u001b[1;32m   2960\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m-> 2961\u001b[0m         X, y \u001b[38;5;241m=\u001b[39m \u001b[43mcheck_X_y\u001b[49m\u001b[43m(\u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43my\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mcheck_params\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   2962\u001b[0m     out \u001b[38;5;241m=\u001b[39m X, y\n\u001b[1;32m   2964\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m no_val_X \u001b[38;5;129;01mand\u001b[39;00m check_params\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mensure_2d\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mTrue\u001b[39;00m):\n",
      "File \u001b[0;32m~/my_ml_education/venv/lib/python3.13/site-packages/sklearn/utils/validation.py:1370\u001b[0m, in \u001b[0;36mcheck_X_y\u001b[0;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_writeable, force_all_finite, ensure_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)\u001b[0m\n\u001b[1;32m   1364\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[1;32m   1365\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mestimator_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m requires y to be passed, but the target y is None\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m   1366\u001b[0m     )\n\u001b[1;32m   1368\u001b[0m ensure_all_finite \u001b[38;5;241m=\u001b[39m _deprecate_force_all_finite(force_all_finite, ensure_all_finite)\n\u001b[0;32m-> 1370\u001b[0m X \u001b[38;5;241m=\u001b[39m \u001b[43mcheck_array\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m   1371\u001b[0m \u001b[43m    \u001b[49m\u001b[43mX\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1372\u001b[0m \u001b[43m    \u001b[49m\u001b[43maccept_sparse\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43maccept_sparse\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1373\u001b[0m \u001b[43m    \u001b[49m\u001b[43maccept_large_sparse\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43maccept_large_sparse\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1374\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdtype\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdtype\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1375\u001b[0m \u001b[43m    \u001b[49m\u001b[43morder\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43morder\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1376\u001b[0m \u001b[43m    \u001b[49m\u001b[43mcopy\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcopy\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1377\u001b[0m \u001b[43m    \u001b[49m\u001b[43mforce_writeable\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mforce_writeable\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1378\u001b[0m \u001b[43m    \u001b[49m\u001b[43mensure_all_finite\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mensure_all_finite\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1379\u001b[0m \u001b[43m    \u001b[49m\u001b[43mensure_2d\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mensure_2d\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1380\u001b[0m \u001b[43m    \u001b[49m\u001b[43mallow_nd\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mallow_nd\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1381\u001b[0m \u001b[43m    \u001b[49m\u001b[43mensure_min_samples\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mensure_min_samples\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1382\u001b[0m \u001b[43m    \u001b[49m\u001b[43mensure_min_features\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mensure_min_features\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1383\u001b[0m \u001b[43m    \u001b[49m\u001b[43mestimator\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mestimator\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1384\u001b[0m \u001b[43m    \u001b[49m\u001b[43minput_name\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mX\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1385\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1387\u001b[0m y \u001b[38;5;241m=\u001b[39m _check_y(y, multi_output\u001b[38;5;241m=\u001b[39mmulti_output, y_numeric\u001b[38;5;241m=\u001b[39my_numeric, estimator\u001b[38;5;241m=\u001b[39mestimator)\n\u001b[1;32m   1389\u001b[0m check_consistent_length(X, y)\n",
      "File \u001b[0;32m~/my_ml_education/venv/lib/python3.13/site-packages/sklearn/utils/validation.py:931\u001b[0m, in \u001b[0;36mcheck_array\u001b[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_writeable, force_all_finite, ensure_all_finite, ensure_non_negative, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)\u001b[0m\n\u001b[1;32m    927\u001b[0m pandas_requires_conversion \u001b[38;5;241m=\u001b[39m \u001b[38;5;28many\u001b[39m(\n\u001b[1;32m    928\u001b[0m     _pandas_dtype_needs_early_conversion(i) \u001b[38;5;28;01mfor\u001b[39;00m i \u001b[38;5;129;01min\u001b[39;00m dtypes_orig\n\u001b[1;32m    929\u001b[0m )\n\u001b[1;32m    930\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mall\u001b[39m(\u001b[38;5;28misinstance\u001b[39m(dtype_iter, np\u001b[38;5;241m.\u001b[39mdtype) \u001b[38;5;28;01mfor\u001b[39;00m dtype_iter \u001b[38;5;129;01min\u001b[39;00m dtypes_orig):\n\u001b[0;32m--> 931\u001b[0m     dtype_orig \u001b[38;5;241m=\u001b[39m \u001b[43mnp\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mresult_type\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mdtypes_orig\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    932\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m pandas_requires_conversion \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;28many\u001b[39m(d \u001b[38;5;241m==\u001b[39m \u001b[38;5;28mobject\u001b[39m \u001b[38;5;28;01mfor\u001b[39;00m d \u001b[38;5;129;01min\u001b[39;00m dtypes_orig):\n\u001b[1;32m    933\u001b[0m     \u001b[38;5;66;03m# Force object if any of the dtypes is an object\u001b[39;00m\n\u001b[1;32m    934\u001b[0m     dtype_orig \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mobject\u001b[39m\n",
      "\u001b[0;31mDTypePromotionError\u001b[0m: The DType <class 'numpy.dtypes.DateTime64DType'> could not be promoted by <class 'numpy.dtypes.Int64DType'>. This means that no common DType exists for the given inputs. For example they cannot be stored in a single array unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>, <class 'numpy.dtypes.DateTime64DType'>, <class 'numpy.dtypes.Int64DType'>)"
     ]
    }
   ],
   "source": [
    "# Ваш код здесь\n",
    "get_auc_lr_valid(train_df,y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Будем считать эту модель нашей первой отправной точкой (baseline). Для построения модели для прогноза на тестовой выборке **необходимо обучить модель заново уже на всей обучающей выборке** (пока наша модель обучалась лишь на части данных), что повысит ее обобщающую способность:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# функция для записи прогнозов в файл\n",
    "def write_to_submission_file(\n",
    "    predicted_labels, out_file, target=\"target\", index_label=\"session_id\"\n",
    "):\n",
    "    predicted_df = pd.DataFrame(\n",
    "        predicted_labels,\n",
    "        index=np.arange(1, predicted_labels.shape[0] + 1),\n",
    "        columns=[target],\n",
    "    )\n",
    "    predicted_df.to_csv(out_file, index_label=index_label)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Обучите модель на всей выборке, сделайте прогноз для тестовой выборки и сделайте посылку в соревновании**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ваш код здесь"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Если вы выполните эти действия и загрузите ответ на [странице](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) соревнования, то воспроизведете первый бенчмарк \"Logit\".\n",
    "\n",
    "### 3. Улучшение модели, построение новых признаков"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Создайте такой признак, который будет представлять собой число вида ГГГГММ от той даты, когда проходила сессия, например 201407 -- 2014 год и 7 месяц. Таким образом, мы будем учитывать помесячный [линейный тренд](http://people.duke.edu/~rnau/411trend.htm) за весь период предоставленных данных."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ваш код здесь"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Добавьте новый признак, предварительно отмасштабировав его с помощью `StandardScaler`, и снова посчитайте ROC AUC на отложенной выборке."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ваш код здесь"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Добавьте два новых признака: start_hour и morning.**\n",
    "\n",
    "Признак `start_hour` – это час в который началась сессия (от 0 до 23), а бинарный признак `morning` равен 1, если сессия началась утром и 0, если сессия началась позже (будем считать, что утро это если `start_hour равен` 11 или меньше).\n",
    "\n",
    "**Посчитйте ROC AUC на отложенной выборке для выборки с:**\n",
    "- сайтами, `start_month` и `start_hour`\n",
    "- сайтами, `start_month` и `morning`\n",
    "- сайтами, `start_month`, `start_hour` и `morning`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ваш код здесь"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Подбор коэффицициента регуляризации\n",
    "\n",
    "Итак, мы ввели признаки, которые улучшают качество нашей модели по сравнению с первым бейслайном. Можем ли мы добиться большего значения метрики? После того, как мы сформировали обучающую и тестовую выборки, почти всегда имеет смысл подобрать оптимальные гиперпараметры -- характеристики модели, которые не изменяются во время обучения. Например, на 3 неделе вы проходили решающие деревья, глубина дерева это гиперпараметр, а признак, по которому происходит ветвление и его значение -- нет. В используемой нами логистической регрессии веса каждого признака изменяются и во время обучения находится их оптимальные значения, а коэффициент регуляризации остается постоянным. Это тот гиперпараметр, который мы сейчас будем оптимизировать.\n",
    "\n",
    "Посчитайте качество на отложенной выборке с коэффициентом регуляризации, который по умолчанию `C=1`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ваш код здесь"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Постараемся побить этот результат за счет оптимизации коэффициента регуляризации. Возьмем набор возможных значений C и для каждого из них посчитаем значение метрики на отложенной выборке.\n",
    "\n",
    "Найдите `C` из `np.logspace(-3, 1, 10)`, при котором ROC AUC на отложенной выборке максимален. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ваш код здесь"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Наконец, обучите модель с найденным оптимальным значением коэффициента регуляризации и с построенными признаками `start_hour`, `start_month` и `morning`. Если вы все сделали правильно и загрузите это решение, то повторите второй бенчмарк соревнования."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ваш код здесь"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
