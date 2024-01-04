#include "tree_serializer.h"


void postprocess_node_map(  map<int,CTNode*> node_map, 
                            MatrixXd *data_store, 
                            Document& document) {

  Value nodeList(kArrayType);
  
  CTNode* curr = nullptr;
  curr = node_map[-1];
  
  
  Value root(kObjectType);
  Value nullValue;
  
  nullValue.SetNull();

  // generic root header
  root.AddMember("node_id", 0, document.GetAllocator());
  root.AddMember("node_val_idx", nullValue, document.GetAllocator());
  
  // create children array
  Value children(kArrayType);
  for (auto &c : curr->children)
    children.PushBack(c, document.GetAllocator());
  root.AddMember("children", children, document.GetAllocator());

  root.AddMember("data_refs", nullValue, document.GetAllocator());

  root.AddMember("param_refs", nullValue, document.GetAllocator());
  nodeList.PushBack(root, document.GetAllocator());
  
  for (auto &n : node_map) {
    if (n.first == -1) 
      continue;
    
    curr = n.second;

    
    if (curr->data == nullptr) {
      // printf("CLEAR");
      Value node(kObjectType);
      node.AddMember("node_id", curr->_id + 1, document.GetAllocator());
      node.AddMember("node_val_idx", curr->_id + 2, document.GetAllocator());
      Value curr_children(kArrayType);
      for (auto &c : curr->children)
        curr_children.PushBack(c, document.GetAllocator());
      node.AddMember("children", curr_children, document.GetAllocator());
      node.AddMember("data_refs", nullValue, document.GetAllocator());
      nodeList.PushBack(node, document.GetAllocator());
    } else {
      if (curr->children.size()  != 0) {
        // cout << "FIRE";
        continue;
      }
      Value node(kObjectType);
      // printf("%lu\n", (curr->val - &data_store[0]) / sizeof(curr->val));
      // cout << curr->val - data_store << '\n'<< endl;
      node.AddMember("node_id", curr->_id + 1 , document.GetAllocator());
      node.AddMember("node_val_idx", (curr->val - &data_store[0]) / sizeof(curr->val), document.GetAllocator());
      Value curr_data(kArrayType);
      for (auto &d : *(curr->data))
        curr_data.PushBack(d, document.GetAllocator());
      node.AddMember("children", nullValue, document.GetAllocator());
      node.AddMember("data_refs", curr_data, document.GetAllocator());
      nodeList.PushBack(node, document.GetAllocator());
    }
  }

  document.AddMember("node_list", nodeList, document.GetAllocator());
  // CTNode* curr = nullptr;
  // for (auto &n : node_map) {
  //   if (n.first == -1)
  //     continue;
  //   curr = n.second;
  //   Value
  //   if isInteriorNode(curr)
  // }
}