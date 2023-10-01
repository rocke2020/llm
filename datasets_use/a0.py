from datasets import load_dataset_builder, load_dataset
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
ic.lineWrapWidth = 120
import os, sys
sys.path.append(os.path.abspath('.'))
from utils.log_util import logger
import psutil


cache_dir = '/mnt/nas1/huggingface/cache'
wikipedia_dir = '/mnt/nas1/huggingface/wikipedia/20230601/en'
wikitext_dir = '/mnt/nas1/huggingface/wikitext'
wikitext_103_raw_dir = wikitext_dir + '/wikitext-103-raw-v1'
wikitext_103_raw_train_dir = wikitext_103_raw_dir + '/train'
wikitext_103_dir = wikitext_dir + '/wikitext-103-v1'
wikitext_103_train_dir = wikitext_103_dir + '/train'


def load_wikipedia():
    """  """
    train_files = [f'{wikipedia_dir}/train-00{i:02d}-of-0083.parquet' for i in range(1, 4)]
    ic(train_files[0])
    data_files = {'train': train_files}
    dataset_args = {
        # "keep_linebreaks": False,  # only for text input format
        "split": 'train',
        "cache_dir": cache_dir,
    }
    wiki = load_dataset('parquet', data_files=data_files, **dataset_args)
    logger.info(wiki)
    for item in wiki:
        print(item)
        break


def load_wikitext_2_raw_v1(offline=True, verbose=True):
    """  """
    wikitext_2_raw_v1_dir = '/mnt/nas1/huggingface/wikitext/wikitext-2-raw-v1'
    logger.info('load_wikitext_2_raw_v1')
    if offline:
        data_files = {
            'train': wikitext_2_raw_v1_dir + '/train/' + '0000.parquet',
            'test': wikitext_2_raw_v1_dir + '/test/' + '0000.parquet',
            'validation': wikitext_2_raw_v1_dir + '/validation/' + '0000.parquet',
        }
        raw_datasets = load_dataset(
            'parquet',
            data_files=data_files,
            cache_dir=cache_dir,
        )
    else:
        raw_datasets = load_dataset(
            'wikitext',
            'wikitext-2-raw-v1',
            cache_dir=cache_dir,
        )
    logger.info(raw_datasets)
    if verbose:
        train_dataset = raw_datasets['train']
        count = 0
        for item in train_dataset:
            logger.info(item)
            count += 1
            if count > 10:
                break
    return raw_datasets


def load_wikitext_103():
    """
    raw
    23-09-30 15:15:48 a0.py 72: {'text': ''}
    23-09-30 15:15:48 a0.py 72: {'text': ' = Valkyria Chronicles III = \n'}
    23-09-30 15:15:48 a0.py 72: {'text': ''}
    23-09-30 15:15:48 a0.py 72: {'text': ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . \n'}
    23-09-30 15:15:48 a0.py 72: {'text': " The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . \n"}
    23-09-30 15:15:48 a0.py 72: {'text': " It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 . \n"}
    23-09-30 15:15:48 a0.py 72: {'text': ''}
    23-09-30 15:15:48 a0.py 72: {'text': ' = = Gameplay = = \n'}
    23-09-30 15:15:48 a0.py 72: {'text': ''}
    23-09-30 15:15:48 a0.py 72: {'text': " As with previous Valkyira Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces . Stories are told through comic book @-@ like panels with animated character portraits , with characters speaking partially through voiced speech bubbles and partially through unvoiced text . The player progresses through a series of linear missions , gradually unlocked as maps that can be freely scanned through and replayed as they are unlocked . The route to each story location on the map varies depending on an individual player 's approach : when one option is selected , the other is sealed off to the player . Outside missions , the player characters rest in a camp , where units can be customized and character growth occurs . Alongside the main story missions are character @-@ specific sub missions relating to different squad members . After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game . There are also love simulation elements related to the game 's two main heroines , although they take a very minor role . \n"}

    non raw
    23-09-30 15:35:24 a0.py 89: {'text': ''}
    23-09-30 15:35:24 a0.py 89: {'text': ' = Valkyria Chronicles III = \n'}
    23-09-30 15:35:24 a0.py 89: {'text': ''}
    23-09-30 15:35:24 a0.py 89: {'text': ' Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " <unk> Raven " . \n'}
    23-09-30 15:35:24 a0.py 89: {'text': " The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer <unk> Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . \n"}
    23-09-30 15:35:24 a0.py 89: {'text': " It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 . \n"}
    23-09-30 15:35:24 a0.py 89: {'text': ''}
    23-09-30 15:35:24 a0.py 89: {'text': ' = = Gameplay = = \n'}
    23-09-30 15:35:24 a0.py 89: {'text': ''}
    23-09-30 15:35:24 a0.py 89: {'text': " As with previous <unk> Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces . Stories are told through comic book @-@ like panels with animated character portraits , with characters speaking partially through voiced speech bubbles and partially through unvoiced text . The player progresses through a series of linear missions , gradually unlocked as maps that can be freely scanned through and replayed as they are unlocked . The route to each story location on the map varies depending on an individual player 's approach : when one option is selected , the other is sealed off to the player . Outside missions , the player characters rest in a camp , where units can be customized and character growth occurs . Alongside the main story missions are character @-@ specific sub missions relating to different squad members . After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game . There are also love simulation elements related to the game 's two main heroines , although they take a very minor role . \n"}
    """
    train_files = [f'{wikitext_103_raw_train_dir}/00{i:02d}.parquet' for i in range(0, 2)]
    data_files = {
        'train': train_files,
        'test': wikitext_103_raw_dir + '/test/' + '0000.parquet',
        'validation': wikitext_103_raw_dir + '/validation/' + '0000.parquet',
    }
    dataset_args = {
        # "keep_linebreaks": False,  # only for text input format
        "split": 'train',
    }
    mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    raw_datasets = load_dataset(
        'parquet',
        data_files=data_files,
        cache_dir=cache_dir,
        # **dataset_args
    )
    logger.info(raw_datasets.keys())
    train_dataset = raw_datasets['train']
    count = 0
    for item in train_dataset:
        logger.info(item)
        count += 1
        if count > 10:
            break
    
    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    # 130 MB about 0.5 of saved parquet file size.
    logger.info(f"RAM memory used: {(mem_after - mem_before)} MB")  


if __name__ == "__main__":
    # load_wikipedia()
    load_wikitext_2_raw_v1()
    # load_wikitext_103()
    logger.info('end')