## Project Description
**This capstone project is instructed by Professor Abha Belorkar and teaching assistant Sam Black at [Temple University’s College of Science and Technology](https://cst.temple.edu/)**

This project adapts ideas from the [I’m Something of a Painter Myself Kaggle Competition](https://www.kaggle.com/c/gan-getting-started/overview) to develop machine learning models that can effectively convert input photos to Monet-style paintings and potentially styles of other artists.

## Interactive Dashboard Demo 
**Note**: This is a beta release

Heroku: https://salty-bastion-21751.herokuapp.com/ 

[click here for more information about the dashboard](https://github.com/WayneJWZLemon/FlaskGANDemo)

## Documentation List
* Link to the [Project Charter Document](https://github.com/WayneJWZLemon/CIS4496-GANs-Project/blob/main/Docs/Project_Charter.md)
* Link to the [Bimonthly Report #1](https://github.com/WayneJWZLemon/CIS4496-GANs-Project/blob/main/Docs/Bimonthly-Report-%231.md)
* Link to the [Bimonthly Report #2](https://github.com/WayneJWZLemon/CIS4496-GANs-Project/blob/main/Docs/Bimonthly-Report-%232.md)
* Link to the [Data Report](https://github.com/WayneJWZLemon/CIS4496-GANs-Project/blob/main/Docs/Data-Report.md)
* Link to the [Model Report](https://github.com/WayneJWZLemon/CIS4496-GANs-Project/blob/main/Docs/Model-Report.md)

## Additional Data Acquisition
We extracted paintings of artists such as Edgar Degas, Pablo Picasso, Rembrandt Harmenszoon van Rijn, Alfred Sisley, and Vincent van Gogh
Dutch painter from [WikiArt](https://www.wikiart.org/). After all images are crawled, we used the Pillow package to resize all images to 256 by 256, then ran the resulting image set through a TFRecords creation script found on Kaggle (https://www.kaggle.com/code/cdeotte/how-to-create-tfrecords/notebook).
* https://www.kaggle.com/datasets/techiewaynezheng/degastfrecs
* https://www.kaggle.com/datasets/techiewaynezheng/picassotfrecs
* https://www.kaggle.com/datasets/techiewaynezheng/rembrandttfrecs
* https://www.kaggle.com/datasets/techiewaynezheng/sisleytfrecs
* https://www.kaggle.com/datasets/techiewaynezheng/vangogh-paintings


Here is the sample script we used to crawl the images (one can easily change the artist's name in the script to the one desired).
```python
from bs4 import BeautifulSoup
import urllib.request
import requests

## Return the html file of the Monet Wikiart page
fp = urllib.request.urlopen("https://www.wikiart.org/en/claude-monet/all-works/text-list")
urlbytes = fp.read()
urlstr = urlbytes.decode("utf8")

## By inspecting the html structure, all the Monet paintings are listed in a table form with the "a" tag
## we need to get all "a" tags and retreive the corresponding href attribute to go to the actual image page
## There are some href value that contains link to other social medias and some values are just None
## To make sure we got the actual image link, we use the "/en/claude-monet/" to check the string
soup = BeautifulSoup(urlstr, "html.parser")
hrefList = []
for link in soup.find_all("a"):
    if(link.get("href") is not None and link.get("href").startswith("/en/claude-monet/")):
        hrefList.append(link.get("href"))


l = len(hrefList)
for i in range(l):
    fp = urllib.request.urlopen("https://wikiart.org/"+hrefList[i])
    urlbytes = fp.read()
    urlstr = urlbytes.decode("utf8")
    soup = BeautifulSoup(urlstr, "html.parser")
    for link in soup.find_all("img"):
        if(link.get("itemprop")=="image"):
            with open('crawled_images/'+link.get("title")+".jpg", 'wb') as handle:
                response = requests.get(link.get("src"), stream=True)
                if not response.ok:
                    print(response)
                    break
                for block in response.iter_content(1024):
                    if not block:
                        break
                    handle.write(block)
            break
```
