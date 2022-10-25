void gf(int q,int p)
{
	int i,j;
	 for (j=1;j<=1000;j++) if (abs(i=(long long)q*j%p)<=1000)
	{
		cerr<<i<<"/"<<j<<endl;
		return;
	}
	cerr<<"Not find"<<endl;
}
