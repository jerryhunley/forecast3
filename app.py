
# (Previous code remains unchanged above...)

                st.subheader("📈 Time to Contact + Recent Conversion Trends")

                # [previous content remains intact here]

                else:
                    st.info("Not enough recent data to display conversion trends.")

                # --- Study-Wide Forecast Summary ---
                st.header("📊 Study-Wide Forecast Summary")
                st.markdown("Analyze projected performance, pacing vs. goal, and CPQL efficiency over time")

                total_icf_goal = st.number_input("Total ICF Goal for Study", min_value=1, value=150)
                current_icfs = referrals['Signed ICF'].sum()
                remaining_icfs = total_icf_goal - current_icfs

                st.write(f"**Total Signed ICFs to Date:** {current_icfs}")
                st.write(f"**ICFs Remaining to Goal:** {remaining_icfs}")

                if 'Referral Date' in referrals.columns:
                    referrals['Month'] = referrals['Referral Date'].dt.to_period('M').dt.start_time
                    referrals['Week'] = referrals['Referral Date'].dt.to_period('W').apply(lambda r: r.start_time)

                    monthly_grouped = referrals.groupby('Month').agg(
                        Referrals=('Referral Date', 'count'),
                        ICFs=('Signed ICF', 'sum')
                    ).reset_index()
                    monthly_grouped['Cumulative ICFs'] = monthly_grouped['ICFs'].cumsum()

                    weekly_grouped = referrals.groupby('Week').agg(
                        Referrals=('Referral Date', 'count'),
                        ICFs=('Signed ICF', 'sum')
                    ).reset_index()
                    weekly_grouped['Cumulative ICFs'] = weekly_grouped['ICFs'].cumsum()

                    cpql_input = st.number_input("Estimated Cost per Qualified Lead ($)", value=75)
                    monthly_grouped['Spend ($)'] = monthly_grouped['Referrals'] * cpql_input
                    monthly_grouped['Cost per ICF ($)'] = monthly_grouped.apply(lambda row: row['Spend ($)'] / row['ICFs'] if row['ICFs'] > 0 else None, axis=1)
                    weekly_grouped['Spend ($)'] = weekly_grouped['Referrals'] * cpql_input
                    weekly_grouped['Cost per ICF ($)'] = weekly_grouped.apply(lambda row: row['Spend ($)'] / row['ICFs'] if row['ICFs'] > 0 else None, axis=1)

                    weekly_grouped['ICFs % Change'] = weekly_grouped['ICFs'].pct_change().round(3) * 100
                    weekly_grouped['Referrals % Change'] = weekly_grouped['Referrals'].pct_change().round(3) * 100

                    trailing_window = st.slider("Trailing Window (days)", min_value=14, max_value=120, value=60)
                    recent_referrals = referrals[referrals['Referral Date'] >= pd.Timestamp.now() - pd.Timedelta(days=trailing_window)]
                    trailing_icf_rate = recent_referrals['Signed ICF'].sum() / (trailing_window / 30)

                    months_needed = int(np.ceil(remaining_icfs / trailing_icf_rate)) if trailing_icf_rate > 0 else None
                    projected_completion = (datetime.today() + pd.DateOffset(months=months_needed)).date() if months_needed else "Not enough data"

                    st.write(f"📈 **Projected ICF Completion Rate (last {trailing_window} days):** {trailing_icf_rate:.1f} ICFs/month")
                    st.write(f"📅 **Projected Study Completion Date:** {projected_completion}")

                    if isinstance(projected_completion, datetime.date):
                        days_remaining = (projected_completion - datetime.today().date()).days
                        st.metric(label="⏳ Days Remaining to Completion", value=f"{days_remaining} days")

                    st.markdown("### 📆 Monthly ICF & Cost Trends")
                    st.line_chart(monthly_grouped.set_index('Month')[['ICFs', 'Cumulative ICFs']])
                    st.line_chart(monthly_grouped.set_index('Month')[['Spend ($)', 'Cost per ICF ($)']])

                    st.markdown("### 📅 Weekly ICF & Referral Pacing")
                    st.line_chart(weekly_grouped.set_index('Week')[['ICFs', 'Referrals']])
                    st.markdown("**Week-over-Week % Change**")
                    st.dataframe(weekly_grouped[['Week', 'ICFs % Change', 'Referrals % Change']].dropna(), use_container_width=True)

                    st.markdown("### 🎯 Weekly Goal Line Overlay")
                    weeks_needed = int(np.ceil(total_icf_goal / weekly_grouped['ICFs'].mean())) if weekly_grouped['ICFs'].mean() > 0 else 12
                    target_line = pd.Series([total_icf_goal / weeks_needed] * len(weekly_grouped), index=weekly_grouped['Week'])
                    goal_df = pd.DataFrame({
                        'Actual ICFs': weekly_grouped['ICFs'].values,
                        'Goal Line': target_line.values
                    }, index=weekly_grouped['Week'])
                    st.line_chart(goal_df)

                    st.markdown("### 🔻 Under-Target Weeks")
                    under_target = weekly_grouped[weekly_grouped['ICFs'] < target_line.values]
                    st.dataframe(under_target[['Week', 'ICFs']], use_container_width=True)

                    csv_export = monthly_grouped.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📁 Download Study-Wide Forecast (CSV)",
                        data=csv_export,
                        file_name="study_forecast_trends.csv",
                        mime="text/csv"
                    )
