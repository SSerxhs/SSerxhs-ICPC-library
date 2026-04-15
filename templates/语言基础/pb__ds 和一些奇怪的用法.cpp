#pragma GCC optimize("Ofast")
#pragma GCC target("popcnt","sse3","sse2","sse","avx","sse4","sse4.1","sse4.2","ssse3","f16c","fma","avx2","xop","fma4")
#pragma GCC optimize("inline","fast-math","unroll-loops","no-stack-protector")
#include "bits/stdc++.h"
#include "ext/pb_ds/assoc_container.hpp"
#include "ext/pb_ds/tree_policy.hpp" //balanced tree
#include "ext/pb_ds/hash_policy.hpp" //hash table
#include "ext/pb_ds/priority_queue.hpp" //priority_queue
using namespace __gnu_pbds;
using namespace std;
template <typename T> using rbt = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
cc_hash_table<string, int>mp1;//拉链法
gp_hash_table<string, int>mp2;//查探法
rbt<int> s1, s2;//注意是不可重的
//null_type无映射(低版本g++为null_mapped_type)
//less<int>从小到大排序
//插入t.insert();
//删除t.erase();
//求有多少个数比 k 小:t.order_of_key(k);
//求树中第 k+1 小:t.find_by_order(k);
//a.join(b) b并入a，前提是两棵树的 key 的取值范围不相交，b 会清空但迭代器没事，如不满足会抛出异常。我听说复杂度是线性？？？
//a.split(v,b) key 小于等于 v 的元素属于 a，其余的属于 b
template <typename T> using heap = __gnu_pbds::priority_queue<T, greater<T>, pairing_heap_tag>;
//join(priority_queue &other)  //合并两个堆,other会被清空
//split(Pred prd,priority_queue &other)  //分离出两个堆
//modify(point_iterator it,const key)  //修改一个节点的值
int main()
{
	__builtin_clz();//前导 0
	__builtin_ctz();//后面的 0
	ios::sync_with_stdio(0); cin.tie(0);
	mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
	cout << fixed << setprecision(15);
	rbtree::iterator it;
	string s = "abc", t = "dabce";
	boyer_moore_horspool_searcher S(all(s));
	if (search(all(t), S) != t.end())
	{
		cout << "find\n";
	}
	uniform_real_distribution<> a(1, 2);
	numeric_limits<int>::max();
}

