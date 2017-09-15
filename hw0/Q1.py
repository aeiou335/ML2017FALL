import sys

f = open(sys.argv[1], 'r');
Q1_txt = open("Q1.txt", 'w')

t = list(f)

text = t[0].split(" ")

word_list = []
count_list = []

text[len(text)-1] = text[len(text)-1].split('\n')[0]
found = False
for thisword in text:
	count = text.count(thisword)
	if count > 1:
		for word in word_list:
			if word == thisword:
				found = True
				break
	if found == False:
		word_list.append(thisword)
		count_list.append(count)
	found = False

for index in range(len(word_list)):
	Q1_txt.write('{} {} {}\n'.format(word_list[index], index, count_list[index]))

f.close()
Q1_txt.close()