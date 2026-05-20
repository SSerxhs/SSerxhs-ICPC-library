#include "bits/stdc++.h"
using namespace std;
bitset<10> f(12);
char s2[] = "100101";
bitset<10> g(s2);
string s = "100101";//reverse 了
bitset<10> h(s);
int main()
{
	for (int i = 0; i <= 9; i++) cout << f.test(i); cout << endl;
	for (int i = 0; i <= 9; i++) cout << g.test(i); cout << endl;
	for (int i = 0; i <= 9; i++) cout << h.test(i); cout << endl;
	cout << h << endl;
	foo.count();//1的个数
	foo.flip();//全部翻转
	foo.set();//变1
	foo.reset();//变0
	foo.to_string();
	foo.to_ulong();
	foo.to_ullong();
	foo._Find_first();
	foo._Find_next();
	//位运算：<< 变大，>> 变小
}

