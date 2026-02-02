new Vue({
  el: '#app',
  data: () => ({
    displayAll: false,
    match_groups: [],
    match_to_dir: '',
    match_from_dir: '',
    match_to_dirNum: 0,
    match_from_dirNum: 0,
    match_from_list: [],
    chooseOneDialogShow: false,
    match_from_listShow: false,
    config: {
      match_to_dir: 'F:\\JHenTai_data\\single_Pic\\为什么老师会在这里\\12_work',
      match_from_dir: 'F:\\JHenTai_data\\single_Pic\\为什么老师会在这里\\12_work\\111\\汉化',
      match_from_son: false,
      imgMaxWidth: 200,
      imgMaxHeight: 280,
      similar_threshold: 0.5,
      maskUseCPU: true,
      cv2Align: true,
      maskTempDir: '',
      autoGray: true,
      colorLv: false,
      colorLvBlack: 12,
      colorLvWhite: 230,
      colorLvGray: 0.8,
      blurFilter: false,
      blurFilterRadius: 3,
      blurFilterThreshold: 8,
      USMFilter: false,
      USMFilterQuantity: 65,
      USMFilterRadius: 1,
      USMFilterThreshold: 8,
      UseAction: false,
      UseActionGroup: '',
      UseActionName: '',
      rename_copy: true,
    },
    configTemp: {},
    draggedItem: null,
    btnDisable: false,
    chooseOne: {},
  }),
  methods: {
    resolveImgUrl(path) {
      return `/images/${this.config.imgMaxWidth}x${this.config.imgMaxHeight}/${path}`
    },
    myRequest(url, data) {
      return fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json;charset=utf-8'
        },
        body: JSON.stringify(data)
      }).then(res => {
        if (!res.ok) {
          throw new Error('服务器返回异常状态码：' + res.status);
        }
        return res.json()
      }).then(res => {
        if (res.code !== 200) {
          throw new Error(res['msg'])
        } else {
          return res['data']
        }
      })
    },
    imageDragStart(event, item) {
      event.stopPropagation()
      this.draggedItem = item
    },
    imageDragOver(event) {
      event.stopPropagation()
      event.preventDefault()
    },
    imageDrop(event, idx) {
      event.stopPropagation()
      if (this.draggedItem) {
        this.$set(this.match_groups[idx], 'match', this.draggedItem['name'])
        this.$set(this.match_groups[idx], 'matchPath', this.draggedItem['path'])
        this.$set(this.match_groups[idx], 'matchRatio', 1)
        this.draggedItem = null
      }
    },
    clearMatch(idx) {
      this.$set(this.match_groups[idx], 'match', '')
      this.$set(this.match_groups[idx], 'matchPath', '')
      this.$set(this.match_groups[idx], 'matchRatio', 0)
    },
    saveConfig() {
      localStorage.setItem('auto-ps-config', JSON.stringify(this.config))
    },
    startMatch() {
      this.btnDisable = true
      this.myRequest('/api/img_match', this.config).then(res => {
        this.match_groups = res['match_result']
        this.match_to_dirNum = res['match_from_num']
        this.match_from_dirNum = res['match_dir_num']
        this.match_from_list = res['match_from_list']
      }).catch(err => {
        alert(err)
      }).finally(_ => {
        this.btnDisable = false
      })
      this.match_to_dir = this.config['match_to_dir']
      this.match_from_dir = this.config['match_from_dir']
    },
    startPS() {
      if(this.match_groups_result.length === 0) {
        return window.ELEMENT.Message({
          message: '当前任务列表为空',
          type: 'warning',
          showClose: true,
          duration: 4000
        })
      }
      this.btnDisable = true
      this.myRequest('/api/start_ps', {
        match_list: this.match_groups_result,
        config: this.config
      }).then(_ => {
        window.ELEMENT.Message({
          message: '任务下发成功，本页面不会提示任务进度！',
          type: 'success',
          showClose: true,
          duration: 4000
        })
        return 1
      }).catch(err => {
        alert(err.message || err)
      }).finally(_ => {
        this.btnDisable = false
      })
    },
    startRename() {
      if(this.match_groups_result.length === 0) {
        return window.ELEMENT.Message({
          message: '当前任务列表为空',
          type: 'warning',
          showClose: true,
          duration: 4000
        })
      }
      this.btnDisable = true
      this.myRequest('/api/start_rename', {
        match_list: this.match_groups_result,
        config: this.config
      }).then(_ => {
        return window.ELEMENT.Message({
          message: '成功！',
          type: 'success',
          showClose: true,
          duration: 4000
        })
      }).catch(err => {
        alert(err.message || err)
      }).finally(_ => {
        this.btnDisable = false
      })
    },
    startPS2(no_mask=true) {
      if(this.match_groups_result.length === 0) {
        return window.ELEMENT.Message({
          message: '当前任务列表为空',
          type: 'warning',
          showClose: true,
          duration: 4000
        })
      }
      this.btnDisable = true
      this.myRequest('/api/start_ps', {
        match_list: this.match_groups_result,
        config: this.config,
        no_mask: no_mask
      }).then(_ => {
        window.ELEMENT.Message({
          message: '任务下发成功，本页面不会提示任务进度！',
          type: 'success',
          showClose: true,
          duration: 4000
        })
        return 1
      }).catch(err => {
        alert(err.message || err)
      }).finally(_ => {
        this.btnDisable = false
      })
    },
    startPS3() {
      if(Object.keys(this.chooseOne).length === 0) {
        return window.ELEMENT.Message({
          message: '请选择一张图！',
          type: 'warning',
          showClose: true,
          duration: 4000
        })
      }
      this.btnDisable = true
      this.myRequest('/api/start_ps', {
        match_list: [this.chooseOne],
        config: this.config,
        no_mask: this.$refs.optionTabs.$data.currentName !== '1'
      }).then(_ => {
        window.ELEMENT.Message({
          message: '任务下发成功，本页面不会提示任务进度！',
          type: 'success',
          showClose: true,
          duration: 4000
        })
        return 1
      }).catch(err => {
        alert(err.message || err)
      }).finally(_ => {
        this.btnDisable = false
      })
    },
    chooseOneClick(data) {
      this.chooseOne = {...data}
    }
  },
  computed: {
    match_groups_result() {
      return this.match_groups.filter(item => item['match'] && item['matchRatio'] >= this.config.similar_threshold)
    }
  },
  mounted() {
    let temp = localStorage.getItem('auto-ps-config')
    if (temp) {
      temp = JSON.parse(temp)
      for (const k_ of Object.keys(temp)) {
        this.config[k_] = temp[k_]
      }
    }
  }
})