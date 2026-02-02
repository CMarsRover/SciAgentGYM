import json 
import os 

def abstr_physics_problems(fl_path):
    with open(fl_path,"r",encoding="utf-8")as f:    
        datas = json.load(f) 
    physics_data = [] 
    for cse in datas:
        # if cse.get("isChosen") == True: 
        #     physics_data.append(cse) 
        if cse.get("classification_subfield") == "Physical Chemistry":
            physics_data.append(cse)  
    with open("gpqa_physics_chemistry_problems_mechanics.json","w")as f: 
        json.dump(physics_data,f,indent=4,ensure_ascii=False)
    return physics_data 

if __name__ == "__main__":
    print(abstr_physics_problems("gpqa_questions_converted.json"))

