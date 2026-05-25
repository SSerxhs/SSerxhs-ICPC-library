#include "bits/stdc++.h"
using namespace std;
template<class info, class tag> struct lct
{
	vector<array<int, 2>> c;
	vector<int> f, rev, lz, st;
	vector<info> s, v;
	vector<tag> tg;
#ifdef Rev
	vector<info> rs;
#endif
	lct(int n) :f(n + 1), c(n + 1), s(n + 1), v(n + 1), tg(n + 1), rev(n + 1), lz(n + 1), st(n + 1)
#ifdef Rev
		, rs(n + 1)
#endif
	{ }

	bool nroot(int x) const
	{
		return c[f[x]][0] == x || c[f[x]][1] == x;
	}
	void pushup(int x)
	{
		int lc = c[x][0], rc = c[x][1];
		s[x] = v[x];
#ifdef Rev
		rs[x] = v[x];
#endif
		if (lc)
		{
			s[x] = s[lc] + s[x];
#ifdef Rev
			rs[x] = rs[x] + rs[lc];
#endif
		}
		if (rc)
		{
			s[x] = s[x] + s[rc];
#ifdef Rev
			rs[x] = rs[rc] + rs[x];
#endif
		}
	}
	void swp(int x)
	{
		swap(c[x][0], c[x][1]);
#ifdef Rev
		swap(s[x], rs[x]);
#endif
		rev[x] ^= 1;
	}
	void add(int x, const tag &o)
	{
		s[x] += o; v[x] += o;
#ifdef Rev
		rs[x] += o;
#endif
		if (lz[x]) tg[x] += o; else tg[x] = o, lz[x] = 1;
	}
	void pushdown(int x)
	{
		if (rev[x])
		{
			for (int y : c[x]) if (y) swp(y);
			rev[x] = 0;
		}
		if (lz[x])
		{
			for (int y : c[x]) if (y) add(y, tg[x]);
			lz[x] = 0;
		}
	}
	void zigzag(int x)
	{
		int y = f[x], z = f[y], typ = (c[y][0] == x);
		if (nroot(y)) c[z][c[z][1] == y] = x;
		f[x] = z; f[y] = x;
		if (c[x][typ]) f[c[x][typ]] = y;
		c[y][typ ^ 1] = c[x][typ]; c[x][typ] = y;
		pushup(y);
	}
	void splay(int x)
	{
		int y, tp;
		st[tp = 1] = y = x;
		while (nroot(y)) st[++tp] = y = f[y];
		while (tp) pushdown(st[tp--]);
		for (; nroot(x); zigzag(x)) if (nroot(y = f[x])) zigzag((c[y][0] == x) ^ (c[f[y]][0] == y) ? x : f[x]);
		pushup(x);
	}
	int access(int x)
	{
		int y = 0;
		for (; x; x = f[y = x]) splay(x), c[x][1] = y, pushup(x);
		return y;
	}
	int findroot(int x)//splay 根为树根，splay 维护树根到 x 的链
	{
		access(x); splay(x); pushdown(x);
		while (c[x][0]) pushdown(x = c[x][0]);
		splay(x); return x;
	}
	void split(int x, int y)//x 为树新根，y 为 splay 新根
	{
		makeroot(x); access(y); splay(y);
	}
	void makeroot(int x)//x 为树、splay 新根
	{
		access(x); splay(x); swp(x);
	}
	void modify(int x, const info &o)
	{
		makeroot(x); v[x] = o; pushup(x);
	}
	void modify(int x, int y, const tag &o)
	{
		split(x, y); add(y, o);
	}
	info ask(int x, int y) { split(x, y); return s[y]; }
	bool connected(int x, int y)//注意会改变形态
	{
		makeroot(x); return findroot(y) == x;
	}
	void link(int x, int y)//y 为新根
	{
		if (!connected(x, y)) makeroot(f[x] = y);
	}
	void cut(int x, int y)
	{
		if (connected(x, y))//可能本不连通
		{
			pushdown(x);
			if (c[x][1] == y && !c[y][0] && !c[y][1])//可能连通但无边
			{
				c[x][1] = f[y] = 0;
				pushup(x);
			}
		}
	}
	int lca(int x, int y) { access(x); return access(y); }
	vector<int> res;
	void dfs(int x)
	{
		if (!x) return;
		pushdown(x);
		dfs(c[x][0]); res.push_back(x); dfs(c[x][1]);
	}
	vector<int> get_path(int x, int y)
	{
		res.clear(); split(x, y); dfs(y);
		if (res[0] != x) return { };
		return res;
	}
};
const int N = 2e5 + 5, M = 4e5 + 5;
struct tag
{
	void operator+=(const tag &o) const { }
};
void operator+=(int &x, const tag &o) { x = 0; }
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	int n, m, i, r = 0;
	cin >> n >> m;
	lct<int, tag> s(n * 2), t(n + m);
	for (i = 1; i <= n; i++) s.modify(i + n, 1), t.modify(i, 1);
	int bs = n, ds = n;
	while (m--)
	{
		int op, u, v;
		cin >> op >> u >> v;
		u ^= r; v ^= r;
		if (op == 1)
		{
			if (s.connected(u, v))
			{
				s.modify(u, v, { });
				auto c = t.get_path(u, v);
				for (i = 1; i < c.size(); i++) t.cut(c[i - 1], c[i]);
				++ds;
				for (int x : c) t.link(ds, x);
			}
			else
			{
				s.link(++bs, u);
				s.link(bs, v);
				t.link(++ds, u);
				t.link(ds, v);
			}
		}
		else
		{
			if (!s.connected(u, v))
			{
				cout << "-1\n";
				continue;
			}
			r = op == 2 ? s.ask(u, v) : t.ask(u, v);
			cout << r << '\n';
		}
	}
}
