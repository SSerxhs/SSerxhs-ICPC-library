namespace chordal_graph//下标从 1 开始
{
	const int N=1e5+2;//点数
	bool ed[N];
	vector<int> e[N];
	int n;
	void init(const vector<pair<int,int>> &edges)
	{
		n=0;
		for (auto [u,v]:edges) n=max({n,u,v});
		for (int i=1;i<=n;i++) e[i].clear();
		for (auto [u,v]:edges) e[u].push_back(v),e[v].push_back(u);
	}
	vector<int> perfect_seq(const vector<pair<int,int>> &edges)//MCS
	{
		init(edges);
		static int d[N];
		static vector<int> buc[N];
		int i,mx=0;
		memset(d+1,0,n*sizeof d[0]);
		memset(ed+1,0,n*sizeof ed[0]);
		for (i=1;i<=n;i++) buc[i].clear();
		buc[0].resize(n);
		iota(all(buc[0]),1);
		vector<int> r(n);
		for (i=n-1;i>=0;i--)
		{
			int u=0;
			while (!u)
			{
				while (buc[mx].size()) if (ed[buc[mx].back()]) buc[mx].pop_back();
				else
				{
					ed[u=buc[mx].back()]=1;
					buc[mx].pop_back();
					goto yes;
				}
				--mx;
			}
			yes:;
			r[i]=u;
			for (int v:e[u]) if (!ed[v]) buc[++d[v]].push_back(v),mx=max(mx,d[v]);
		}
		return r;
	}
	bool check_perfect_seq(vector<int> a)
	{
		static bool ee[N];
		static int pos[N];
		memset(ed+1,0,n*sizeof ed[0]);
		memset(ee+1,0,n*sizeof ee[0]);
		for (int i=0;i<n;i++) pos[a[i]]=i;
		for (int u:a)
		{
			int w=0;
			for (int v:e[u]) if (pos[v]>pos[u]&&(!w||pos[v]<pos[w])) w=v;
			if (!w) continue;
			ee[w]=1;
			for (int v:e[w]) ee[v]=1;
			for (int v:e[u]) if (pos[v]>pos[u]&&!ee[v]) return 0;
			ee[w]=0;
			for (int v:e[w]) ee[v]=0;
		}
		return 1;
	}
	bool check_chordal(const vector<pair<int,int>> &edges) {return check_perfect_seq(perfect_seq(edges));}
	vector<int> find_cycle(const vector<pair<int,int>> &edges)//若不是弦图，返回一个无弦环。首尾相连，不重复首点
	{
		auto a=perfect_seq(edges);
		static bool ee[N];
		static int pos[N],pre[N];
		memset(ee+1,0,n*sizeof ee[0]);
		for (int i=0;i<n;i++) pos[a[i]]=i;
		for (int u:a)
		{
			int w=0,vv=0;
			for (int v:e[u]) if (pos[v]>pos[u]&&(!w||pos[v]<pos[w])) w=v;
			if (!w) continue;
			ee[w]=1;
			for (int v:e[w]) ee[v]=1;
			for (int v:e[u]) if (pos[v]>pos[u]&&!ee[v]) {vv=v;break;}
			ee[w]=0;
			for (int v:e[w]) ee[v]=0;
			if (!vv) continue;
			memset(ed+1,0,n*sizeof ed[0]);
			memset(pre+1,0,n*sizeof pre[0]);
			ed[u]=1;
			for (int v:e[u]) if (v!=w&&v!=vv) ed[v]=1;
			queue<int> q;
			q.push(w);pre[w]=-1;
			while (q.size())
			{
				int x=q.front();q.pop();
				if (x==vv) break;
				for (int y:e[x]) if (!ed[y]&&!pre[y]) pre[y]=x,q.push(y);
			}
			if (!pre[vv]) continue;
			vector<int> r;
			for (int x=vv;x!=-1;x=pre[x]) r.push_back(x);
			reverse(all(r));
			r.insert(r.begin(),u);
			return r;
		}
		return { };
	}
	vector<int> color(int _n,const vector<pair<int,int>> &edges)//返回长度为 _n+1。其中 0 无意义
	{
		auto a=perfect_seq(edges);
		reverse(all(a));
		memset(ed+1,0,n*sizeof ed[0]);
		vector<int> r(_n+1);
		for (int u:a)
		{
			for (int v:e[u]) ed[r[v]]=1;
			int x=1;
			while (ed[x]) ++x;
			r[u]=x;
			for (int v:e[u]) ed[r[v]]=0;
		}
        for (int i=n+1;i<=_n;i++) r[i]=1;
		return r;
	}
	vector<int> max_independent(int _n,const vector<pair<int,int>> &edges)//注意有孤立点这种奇怪东西
	{
		auto a=perfect_seq(edges);
		memset(ed+1,0,n*sizeof ed[0]);
		vector<int> r;
		for (int u:a) if (!ed[u])
		{
			r.push_back(u);
			for (int v:e[u]) ed[v]=1;
		}
		for (int i=n+1;i<=_n;i++) r.push_back(i);
		return r;
	}
}
using chordal_graph::check_chordal,chordal_graph::find_cycle,chordal_graph::color,chordal_graph::max_independent;
