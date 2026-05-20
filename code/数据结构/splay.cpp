
template<class info, class tag> struct splay
{
#define _rev
	struct node
	{
		node *c[2], *f;
		int siz;
		info s, v;
		tag t;
		node() :c{ }, f(0), siz(1), s(), v(), t() { }
		node(info x) :c{ }, f(0), siz(1), s(x), v(x), t() { }
		void operator+=(const tag &o)
		{
			s += o; v += o; t += o;
#ifdef _rev
			if (o.rev) swap(c[0], c[1]);
#endif
		}
		void pushup()
		{
			if (c[0]) s = c[0]->s + v, siz = c[0]->siz + 1; else s = v, siz = 1;
			if (c[1]) s = s + c[1]->s, siz += c[1]->siz;
		}
		void pushdown()
		{
			for (auto x : c) if (x) *x += t;
			t = { };
		}
		void zigzag()
		{
			node *y = f, *z = y->f;
			int typ = y->c[0] == this;
			if (z) z->c[z->c[1] == y] = this;
			f = z; y->f = this;
			y->c[typ ^ 1] = c[typ];
			if (c[typ]) c[typ]->f = y;
			c[typ] = y;
			y->pushup();
		}
		void splay(node *tar)//不要在 makeroot 以外调用
		{
			for (node *y = f; y != tar; zigzag(), y = f) if (node *z = y->f; z != tar) (z->c[1] == y ^ y->c[1] == this ? this : y)->zigzag();
			pushup();
		}
		void clear()
		{
			for (node *x : c) if (x) x->clear();
			delete this;
		}
	};
	node *rt;
	void debug()
	{
		map<node *, int> id;
		id[0] = 0; id[rt] = 1;
		int cnt = 1;
		function<void(node *)> out = [&](node *x) {
			if (!x) return;
			for (auto y : x->c) if (!id.count(y)) id[y] = ++cnt;
			cerr << id[x] << ' ' << id[x->c[0]] << ' ' << id[x->c[1]] << ' ' << id[x->f] << ' ' << x->siz << '\n';
			for (auto y : x->c) out(y);
		};
		out(rt);
	}
	node *build(info *a, int n)
	{
		if (n == 0) return 0;
		int m = n - 1 >> 1;
		node *x = new node(a[m]);
		x->c[0] = build(a, m);
		x->c[1] = build(a + m + 1, n - 1 - m);
		for (node *y : x->c) if (y) y->f = x;
		x->pushup();
		return x;
	}
	splay()
	{
		rt = new node;
		rt->c[1] = new node;
		rt->c[1]->f = rt;
		rt->siz = 2;
	}
	int shift;
	splay(info *a, int l, int r)//[l,r)
	{
		shift = l - 1;
		rt = new node;
		rt->c[1] = new node;
		rt->c[1]->f = rt;
		if (l < r)
		{
			rt->c[1]->c[0] = build(a + l, r - l);
			rt->c[1]->c[0]->f = rt->c[1];
		}
		rt->c[1]->pushup();
		rt->pushup();
	}
	void makeroot(node *u, node *tar)
	{
		if (!tar) rt = u;
		u->splay();
	}
	void findnth(int k, node *tar)
	{
		node *x = rt;
		while (1)
		{
			x->pushdown();
			int v = x->c[0] ? x->c[0]->siz : 0;
			if (v + 1 == k) { x->splay(tar); if (!tar) rt = x; return; }
			if (v >= k) x = x->c[0]; else x = x->c[1], k -= v + 1;
		}
	}
	void split(int l, int r)
	{
		assert(1 <= l && r <= rt->siz - 2 && l - 1 <= r);
		findnth(l, 0);
		findnth(r + 2, rt);
	}
#ifdef _rev
	void reverse(int l, int r)
	{
		l -= shift; r -= shift + 1;
		if (l - 1 == r) return;
		assert(1 <= l && l <= r && r <= rt->siz - 2);
		split(l, r);
		*(rt->c[1]->c[0]) += tag(1);
	}
#endif
	void insert(int pos, info x)//insert before pos
	{
		pos -= shift;
		assert(1 <= pos && pos <= rt->siz - 1);
		split(pos, pos - 1);
		rt->c[1]->c[0] = new node(x);
		rt->c[1]->c[0]->f = rt->c[1];
		rt->c[1]->pushup();
		rt->pushup();
	}
	void insert(int pos, info *a, int n)//insert before pos, [1,n]
	{
		pos -= shift;
		assert(1 <= pos && pos <= rt->siz - 1);
		split(pos, pos - 1);
		rt->c[1]->c[0] = build(a, n);
		rt->c[1]->c[0]->f = rt->c[1];
		rt->c[1]->pushup();
		rt->pushup();
	}
	void erase(int pos)
	{
		pos -= shift;
		assert(1 <= pos && pos <= rt->siz - 2);
		split(pos, pos);
		delete rt->c[1]->c[0];
		rt->c[1]->c[0] = 0;
		rt->c[1]->pushup();
		rt->pushup();
	}
	void erase(int l, int r)
	{
		l -= shift;  r -= shift + 1;
		if (l - 1 == r) return;
		assert(1 <= l && l <= r && r <= rt->siz - 2);
		split(l, r);
		rt->c[1]->c[0]->clear();
		rt->c[1]->c[0] = 0;
		rt->c[1]->pushup();
		rt->pushup();
	}
	void modify(int pos, info x)//not checked
	{
		pos -= shift;
		assert(1 <= pos && pos <= rt->siz - 2);
		findnth(pos + 1, 0);
		rt->v = x; rt->pushup();
	}
	void modify(int l, int r, tag w)
	{
		l -= shift; r -= shift + 1;
		if (l - 1 == r) return;
		assert(1 <= l && l <= r && r <= rt->siz - 2);
		split(l, r);
		node *x = rt->c[1]->c[0];
		*x += w;
		rt->c[1]->pushup();
		rt->pushup();
	}
	info ask(int l, int r)
	{
		l -= shift; r -= shift + 1;
		assert(1 <= l && l <= r && r <= rt->siz - 2);
		split(l, r);
		return rt->c[1]->c[0]->s;
	}
	~splay() { rt->clear(); }
#undef _rev
};
struct Q
{
	bool rev;
	Q() :rev(0) { }
	Q(bool c) :rev(c) { }
	void operator+=(const Q &o)
	{
		rev ^= o.rev;
	}
};
struct P
{
	ll s;
	void operator+=(const Q &o) const
	{ }
	P operator+(const P &o) const { return{s + o.s}; }
};


