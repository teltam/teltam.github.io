#### Theme
pandoc --print-highlight-style tango > tango.theme



#### Generate
pandoc -H cf.html --toc --highlight-style tango.theme --resource-path=media -f markdown md/fine-tuning-with-a-4090.md > posts/fine-tuning-with-a-4090.html
