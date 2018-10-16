
plot(timeminutes, sSWNTm-sSWNTm(3500))
hold on
plot(timeminutes, sSWNTmUV-sSWNTmUV(3500))
plot(timeminutes, mSWNTm-mSWNTm(3500))
plot(timeminutes, mSWNTmUV-mSWNTmUV(3500))


plot(timeminutes, MWNTm-MWNTm(3500))

plot(timeminutes, MWNTmUV-MWNTmUV(3500))


legend ('sSWNTmr','sSWNTmrUV','mSWNTmr','mSWNTmrUV','MWNTmr','MWNTmrUV')



hold off


%%
ad = sSWNTmUV-sSWNTmUV(1) - .00.*timeminutes
plot(timeminutes,ad)
