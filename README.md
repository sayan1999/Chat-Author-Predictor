# Whatsapp-chat-author-classifier
# An awesome python application, very simple to install and run
predicts the author of chat or chat-line, a model trainable from whatsapp exportable chat texts
 
# Instructions

1. Open project directory.

2. Export whatsapp chats from your phone. Go through [link](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&ved=2ahUKEwi4kdHC7IPnAhWd6nMBHQmPC54QFjACegQIDRAH&url=https%3A%2F%2Fwww.guidingtech.com%2Fexport-whatsapp-chat-pdf%2F&usg=AOvVaw2GF3qCNfKOnuFi44dyAq6L) for help.

3. Use personal chats and not groupchats preferably. (optional)

4. Create a directory named dataset and put the exported chats inside that 'dataset' directory.

5. Open 'myname.json' and modify "name" value and by replacing "Sayan Dey" with <your name mentioned in your whatsapp account at the time of exporting chats>

7. Automated: Change mode of the file train to executable, run it and wait.
	
In linux terminal:

```bash
	chmod +x train
	./train

```

	Manual: Run the three python script in order: WhatsApp.py, process_dataframe.py, author_classify.py

	Alternative: Run the .ipynb files in jupyter or with IPython in order: WhatsApp.ipynb, process_dataframe.ipynb, author_classify.ipynb

8. Open 'test.ipynb' with jupyter or ipython. 
The sentence variable is a list of strings.

Modify the strings to the text you think someone often says or texts and run it. It might match 
with your person in imagination.