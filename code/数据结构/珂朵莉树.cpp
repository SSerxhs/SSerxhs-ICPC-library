namespace chtholly_tree
{
	using T = int;//可以把 T 修改为任意想要的类型。
	struct node
	{
		int l;
		mutable int r;
		mutable T v;
		int len() const { return r - l + 1; }
		bool operator<(const node &x) const { return l < x.l; }
	};
	void add(const node &a) { }
	void del(const node &a) { }
	class odt : public set<node>
	{
	public:
		typedef odt::iterator iter;
		iter split(int x)
		{
			iter it = lower_bound({x});
			if (it != end() && it->l == x) return it;
			node t = *--it, a = {t.l, x - 1, t.v}, b = {x, t.r, t.v};
			del(*it); add(a); add(b);
			erase(it); insert(a);
			return insert(b).first;
		}
		iter modify(int l, int r, T v)//[l,r]
		{
			iter lt, rt, it;
			rt = r == rbegin()->r ? end() : split(r + 1); lt = split(l);//[lt,rt)
			while (lt != begin() && (it = prev(lt))->v == v) l = (lt = it)->l;
			while (rt != end() && rt->v == v) r = (rt++)->r;
			for (it = lt; it != rt; it++) del(*it);
			add({l, r, v});
			erase(lt, rt);
			return insert({l, r, v}).first;
		}
		T operator[](const int x) const { return prev(upper_bound({x}))->v; }//直接访问单点
		iter find(int x) const { return prev(upper_bound({x})); }//找到对应的线段
	};
}
using chtholly_tree::node, chtholly_tree::odt;
typedef odt::iterator iter;
int main()
{
	odt s;
	s.insert({0, 5, 1}); 	// 先 insert({L,R,x}) 表示整个下标范围和初始值。 左闭右闭。
							// s={1,1,1,1,1,1}
	s.modify(2, 3, 2); 		// 左闭右闭。s={1,1,2,2,1,1}
	for (auto [l, r, v] : s)
	{
		//(l,r,v)=(0,1,1)
		//(l,r,v)=(2,3,2)
		//(l,r,v)=(4,5,1)
	}
}

