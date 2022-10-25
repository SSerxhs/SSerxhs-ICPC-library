template<typename T,int n> struct matrix
{
	#define all(x) (x).begin(),(x).end()
	array<pair<int,T>,n> a;
	matrix(char c='E')
	{
		int i;
		if (c=='E') for (i=0;i<n;i++) a[i]={i,0};
		else assert(0);
	}
	matrix(char c,int x)
	{
		
	}
	matrix operator+(const matrix &o) const
	{
		matrix r;
		int i,j,k;
		for (i=0;i<n;i++)
		{
			auto [x,y]=a[i];
			r.a[i]={o.a[x].first,o.a[x].second+y};
		}
		return r;
	}
};
