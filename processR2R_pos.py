import spacy
import json


# get instructions from training json data file
instructions = []
instruction_lengths = []
with open("R2R_train.json") as f:
	training_data = json.load(f)
	for data in training_data:
		instructions.extend(data["instructions"])

for ins in instructions:
	instruction_lengths.append(len(ins))

print("# instructions", 
	len(instructions), 
	"# avg length of instructions", 
	float(sum(instruction_lengths))/float(len(instructions)))


# get sub instructions
sub_instructions = []
nlp = spacy.load("en_core_web_sm")

def add_sub_instruction(sub_inst, sub_instructions):
	if len(sub_inst) > 0:
		if len(sub_instructions) > 0:
			# e.g. ["continue", "going to"] should be ["continue going to"]
			if len(sub_instructions[-1].split()) == 1: 
				sub_instructions[-1] += sub_inst.strip()
		else:
			sub_instructions.append(sub_inst.strip())

# instructions = ["walk through the door and you'll see a table and then stop"]
for i in instructions:
	i = i.lower()
	doc = nlp(i)
	sub_inst = ''
	for token in doc:
		# print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_)
		
		# ignore punctuation and empty token
		if token.pos_ == 'PUNCT' or len(token.text) == 0:
			continue
		# ignore verb like 'will'
		if token.pos_ == 'VERB' and token.dep_ != 'aux':	
			add_sub_instruction(sub_inst, sub_instructions)
			sub_inst = token.text
		else:
			sub_inst += ' ' + token.text
	# final sub goal
	if len(sub_inst) > 0:
		sub_instructions.append(sub_inst.strip())

print(len(sub_instructions))

# write processed sub instructions to output txt file
# with open("training_subinstructions_pos.txt", 'w') as output:
# 	for i in sub_instructions:
# 		output.write(i + '\n')