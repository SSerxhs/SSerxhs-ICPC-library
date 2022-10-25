template<int n> struct matrix
{
#define all(x) (x).begin(),(x).end()
	typedef unsigned int ui;
	typedef unsigned long long ll;
	array<array<ll,n>,n> a;
	matrix(char c='O')
	{
		int i;
		if (c=='O') for (i=0; i<n; i++) fill(all(a[i]),0);
		else if (c=='E') for (i=0; i<n; i++) fill(all(a[i]),0),a[i][i]=1;
		else assert(0);
	}
	matrix(char c,int x)
	{

	}
	matrix operator+(const matrix &o) const
	{
		matrix r;
		int i,j,k;
		for (k=0; k<n; k++)
		{
			for (i=0; i<n; i++) for (j=0; j<n; j++) r.a[i][j]+=a[i][k]*o.a[k][j];
			if (k==n-1||(k&15)==15) for (i=0; i<n; i++) for (j=0; j<n; j++) r.a[i][j]%=p;
		}
		return r;
	}
	static_assert(numeric_limits<ll>::max()/(p-1)/(p-1)>=17);
};
template<int n> ostream &operator<<(ostream &cout,const matrix<n> &o)
{
	int i,j;
	for (i=0; i<n; i++) for (j=0; j<n; j++) cout<<o.a[i][j]<<" \n"[j+1==n];
	return cout;
}