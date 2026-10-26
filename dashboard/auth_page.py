"""Local analyst sign-in and workspace authentication gate."""

import os
import sqlite3
from pathlib import Path
import streamlit as st
import config


def _start_session(user):
    st.session_state.clear()
    st.session_state['auth_user'] = user
    st.session_state['navigation'] = 'Dashboard'
    st.rerun()


def _login_as_demo_analyst(store):
    demo_email = 'demo.analyst@fraudguard.local'
    demo_pass = 'DemoAnalyst-2026!'
    user = store.authenticate(demo_email, demo_pass)
    if not user:
        try:
            user = store.create_user('Demo Analyst', demo_email, demo_pass)
        except ValueError:
            user = store.authenticate(demo_email, demo_pass)
    if user:
        # Pre-seed sample dataset if none exists
        datasets = store.list_datasets(user['id'])
        if not datasets and os.path.exists(config.SAMPLE_TEST_PATH):
            sample_bytes = Path(config.SAMPLE_TEST_PATH).read_bytes()
            d = store.save_dataset(user['id'], 'Synthetic Demo Transactions', sample_bytes)
            store.set_active_dataset(user['id'], d['id'])
        _start_session(user)


def require_sign_in(store, start_demo=None):
    user = st.session_state.get('auth_user')
    if user:
        return user

    st.header('Sign in to your analyst workspace')
    st.write('Upload transaction data, review suspicious transactions, and manage your analysis history.')
    st.info('An academic fraud-analysis workspace for reviewing transaction datasets. Accounts do not provide access to bank accounts or payment services.')

    main, aside = st.columns([1.3, 1], gap='large')
    with main:
        sign_in_tab, create_tab = st.tabs(['Sign in', 'Create analyst account'])
        with sign_in_tab:
            with st.form('sign_in_form', clear_on_submit=True):
                email = st.text_input('Email', key='login_email')
                password = st.text_input('Password', type='password', key='login_password')
                submitted = st.form_submit_button('Sign in', type='primary', width='stretch')
            if submitted:
                try:
                    authenticated = store.authenticate(email, password)
                    if authenticated:
                        _start_session(authenticated)
                    else:
                        st.error('Email or password is incorrect.')
                except (ValueError, sqlite3.Error):
                    st.error('Unable to sign in. Check your details and try again.')
        with create_tab:
            with st.form('create_account_form', clear_on_submit=True):
                full_name = st.text_input('Full name', key='signup_name')
                email = st.text_input('Email address', key='signup_email')
                password = st.text_input('Create password', type='password', key='signup_password')
                confirmation = st.text_input('Confirm password', type='password', key='signup_confirmation')
                st.caption('Use a password or passphrase with at least 12 characters.')
                submitted = st.form_submit_button('Create analyst account', type='primary', width='stretch')
            if submitted:
                if password != confirmation:
                    st.error('Passwords do not match.')
                else:
                    try:
                        _start_session(store.create_user(full_name, email, password))
                    except ValueError as error:
                        st.error(str(error))
                    except sqlite3.Error:
                        st.error('Unable to create the account. Please try again.')

        st.divider()
        if st.button('Try demo workspace', width='stretch', help='Explore synthetic example data in a separate demo workspace.'):
            if start_demo is not None:
                start_demo()
            else:
                _login_as_demo_analyst(store)
        st.caption('Explore synthetic example data in a separate demo workspace. A personal account is not required.')

    with aside:
        with st.container(border=True):
            st.subheader('Your analyst workspace')
            st.markdown('**01 · Save your data**  \nUpload a transaction CSV once and reopen it from your account.')
            st.markdown('**02 · Compare ML and DL**  \nRun the saved Random Forest, XGBoost, deep classifier, or autoencoder.')
            st.markdown('**03 · Keep the evidence**  \nPrediction results and model details remain available in your analysis history.')
            st.markdown('**04 · Review & Escalate**  \nFlag suspicious transactions, add analyst notes, and track review status.')
        st.caption('Academic fraud analysis using machine learning and deep learning. This project has no bank account or payment access.')
        st.caption('Accounts and work are stored locally on this computer. Email verification and password recovery are unavailable in this academic demonstration.')
    st.stop()


def sign_out():
    st.session_state.clear()
    st.rerun()
