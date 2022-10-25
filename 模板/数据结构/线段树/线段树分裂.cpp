namespace sgt
{
#define ask_kth
	int L=0,R=1e9;
	void set_bound(int l,int r) { L=l; R=r; }
	typedef ll info;
	const info E=0;//找不到会返回 E
	const int N=8e6+5;
#define lc(x) (a[x].lc)
#define rc(x) (a[x].rc)
#define s(x) (a[x].s)
	struct node
	{
		int lc,rc;
		info s;
	};
	node a[N];
	vector<int> id;
	int ids=0,pos,z,y;
	bool fir;
	info tmp;
	int npt()
	{
		int x;
		if (id.size()) x=id.back(),id.pop_back();
		else x=++ids;
		lc(x)=rc(x)=0;
		return x;
	}
	void pushup(int &x)
	{
		if (lc(x)&&rc(x)) s(x)=s(lc(x))+s(rc(x));
		else if (lc(x)) s(x)=s(lc(x));
		else if (rc(x)) s(x)=s(rc(x));
		else id.push_back(x),x=0;
	}
	void insert(int &x,int l,int r)
	{
		if (l==r)
		{
			if (!x) x=npt(),s(x)=tmp;
			else s(x)=s(x)+tmp;
			return;
		}
		if (!x) x=npt();
		int mid=l+r>>1;
		if (pos<=mid)
		{
			insert(lc(x),l,mid);
			if (rc(x)) s(x)=s(lc(x))+s(rc(x)); else s(x)=s(lc(x));
		}
		else
		{
			insert(rc(x),mid+1,r);
			if (lc(x)) s(x)=s(lc(x))+s(rc(x)); else s(x)=s(rc(x));
		}
	}
	void modify(int &x,int l,int r)
	{
		if (!x) x=npt();
		if (l==r)
		{
			s(x)=tmp;
			return;
		}
		int mid=l+r>>1;
		if (pos<=mid)
		{
			insert(lc(x),l,mid);
			if (rc(x)) s(x)=s(lc(x))+s(rc(x)); else s(x)=s(lc(x));
		}
		else
		{
			insert(rc(x),mid+1,r);
			if (lc(x)) s(x)=s(lc(x))+s(rc(x)); else s(x)=s(rc(x));
		}
	}
	int merge(int x1,int x2,int l,int r)
	{
		if (!(x1&&x2)) return x1|x2;
		if (l==r) { s(x1)=s(x1)+s(x2); return x1; }
		int mid=l+r>>1;
		lc(x1)=merge(lc(x1),lc(x2),l,mid);
		rc(x1)=merge(rc(x1),rc(x2),mid+1,r);
		pushup(x1);
		return x1;
	}
	void ask(int x,int l,int r)
	{
		if (!x) return;
		if (z<=l&&r<=y)
		{
			if (fir) tmp=s(x),fir=0; else tmp=tmp+s(x);
			return;
		}
		int mid=l+r>>1;
		if (z<=mid) ask(lc(x),l,mid);
		if (y>mid) ask(rc(x),mid+1,r);
	}
	void split(int &x1,int &x2,int l,int r)
	{
		assert(!x1);
		if (!x2) return;
		if (z<=l&&r<=y) { x1=x2; x2=0; return; }
		x1=npt();
		int mid=l+r>>1;
		if (z<=mid) split(lc(x1),lc(x2),l,mid);
		if (y>mid) split(rc(x1),rc(x2),mid+1,r);
		pushup(x1); pushup(x2);
	}
	info *b;
	void build(int &x,int l,int r)
	{
		x=npt();
		if (l==r) { s(x)=b[l]; return; }
		int mid=l+r>>1;
		build(lc(x),l,mid); build(rc(x),mid+1,r);
		s(x)=s(lc(x))+s(rc(x));
	}
	struct set
	{
		int rt;
		set():rt(0) {}
		set(info *a):rt(0) { b=a; build(rt,L,R); }
		void modify(int p,const info &o) { pos=p; tmp=o; sgt::modify(rt,L,R); }
		void insert(int p,const info &o) { pos=p; tmp=o; sgt::insert(rt,L,R); }
		void join(const set &o) { rt=merge(rt,o.rt,L,R); }
		info ask(int l,int r)
		{
			z=l; y=r; fir=1;
			sgt::ask(rt,L,R);
			return fir?E:tmp;
		}
		set split(int l,int r)
		{
			z=l; y=r; set p;
			sgt::split(p.rt,rt,L,R);
			return p;
		}
#ifdef ask_kth
		int kth(info k)
		{
			int x=rt,l=L,r=R,mid;
			if (k>s(x)) return -1;
			s(0)=0;
			while (l<r)
			{
				mid=l+r>>1;
				if (s(lc(x))>=k) x=lc(x),r=mid;
				else k-=s(lc(x)),x=rc(x),l=mid+1;
			}
			return l;
		}
#endif
	};
#undef lc
#undef rc
#undef s
}
typedef sgt::set tree;