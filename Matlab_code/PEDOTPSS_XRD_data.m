offset = 1;

semilogy(A43,B43+.0004*offset)
hold on
semilogy(A64,B64 + .001*offset)
semilogy(A85,B85 + .002*offset)
semilogy(A106,B106 + .003*offset)
semilogy(A127,B127+ .004*offset)
semilogy(A148,B148 + .005*offset)
semilogy(A169,B169 + .006*offset)
semilogy(A190,B190 + .007*offset)
semilogy(A211,B211+ .008*offset)
semilogy(A232,B232 + .009*offset)

legend('43','64','85','106','127','148','169','190','211','232')




hold off