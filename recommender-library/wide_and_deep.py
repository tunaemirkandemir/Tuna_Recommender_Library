import torch
import torch.nn as nn
from torch.utils.data import DataLoader

'''
input_config = {
    'user_id': {'type': 'categorical', 'num_classes': len(le_user.classes_), 'embedding_dim': 10},
    'item_id': {'type': 'categorical', 'num_classes': len(le_item.classes_), 'embedding_dim': 10},
    'age': {'type': 'numerical'},
    'gender': {'type': 'categorical', 'num_classes': len(le_gender.classes_), 'embedding_dim': 2},
    'category': {'type': 'categorical', 'num_classes': len(le_category.classes_), 'embedding_dim': 5},
    'price': {'type': 'numerical'}
}'''
class ConfigurableWideAndDeep(torch.nn.Module):
    def __init__(self,input_config:dict,user_ID_col :str, item_ID_col:str): #burdaki input bi dictionary
        super(ConfigurableWideAndDeep).__init__()
        self.user_id = user_ID_col
        self.item_id = item_ID_col
        self.embeddings = nn.ModuleDict()
        self.deep_input_size = 0
        self.input_config = input_config

        for feature_name ,feature_info in self.input_config.items():
            if feature_info['type'] == 'categorical' or feature_name == 'user_id' or feature_name == 'item_id':
                self.embeddings[feature_name] = nn.Embedding(feature_info['num_classes'],feature_info['embedding_dim'])
                self.deep_input_size += feature_info['embedding_dim']
            else:
                self.deep_input_size += 1

        self.deep=nn.Sequential(
            nn.Linear(self.deep_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU() 
        )

        self.output = nn.Linear(self.deep_input_size+32,1)   #bu sadece 32,1 olabilir
            

    def forward(self,x):
        deep_input_list =[]
        wide_input_list = []

        for feature_name ,feature_tensor in x.items():
            if feature_name in self.embeddings:
                embedded = self.embeddings[feature_name](feature_tensor)
                deep_input_list.append(embedded)
            else:
                feature_tensor=feature_tensor.float().unsqueeze(1)
                deep_input_list.append(feature_tensor)
            
            wide_input_list.append(feature_tensor)

        deep_input = torch.cat(deep_input_list,1)
        wide_input = torch.cat(wide_input_list,1)
        deep_output =self.deep(deep_input)
        combined_input = torch.cat((wide_input,deep_output),1)
        output = self.output(combined_input)

        

        return output
    
    def train(self,train_loader:DataLoader,loss_func:torch.nn ,optimizer:torch.optim,epochs:int,learning_rate:float,target_col:str): 
        #buraya bir sürü kontrol koymalıyım
        #ve o kontrollere göre learning rate i yerleştirmeliyim
        input_config = self.input_config
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = {col: torch.tensor(batch[col]) for col in batch if col != target_col}
                labels = torch.tensor(batch[target_col])
                outputs = self(inputs)
                loss = loss_func(outputs,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item(
                    
                )

    def evaluate(self,test_loader:DataLoader,loss_func:torch.nn,target_col:str,verbose=True):
        self.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs = {col: torch.tensor(batch[col]) for col in batch if col != target_col}
                labels = torch.tensor(batch[target_col])
                outputs = self(inputs)
                loss = loss_func(outputs,labels)
                test_loss += loss.item()
        print(f"Test Loss: {test_loss / len(test_loader)}")
        return test_loss



