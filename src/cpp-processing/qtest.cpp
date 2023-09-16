#include <queue>
#include <vector>
#include <iostream>
using namespace std;

int main(void) {
  queue<vector<int>> q;
  vector<int> a = {1, 2, 3};
  vector<int> b = {4, 5};
  vector<int> c = {6, 7, 8, 9};
  q.push(a);
  q.push(b);
  q.push(c);
  vector<int> d;
  d = q.front();
  q.pop();
  for (auto &elem: d) {
    cout << elem << ' ';
  }
  cout << endl;
  d = q.front();
  q.pop();
  for (auto &elem: d) {
    cout << elem << ' ';
  }
  cout << endl;
  d = q.front();
  q.pop();
  for (auto &elem: d) {
    cout << elem << ' ';
  }
  cout << endl;
  return 0;
}