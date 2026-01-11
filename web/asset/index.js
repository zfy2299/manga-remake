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
    configDialogShow: false,
    match_from_listShow: false,
    config: {
      match_to_dir: 'F:\\JHenTai_data\\single_Pic\\为什么老师会在这里\\12_work',
      match_from_dir: 'F:\\JHenTai_data\\single_Pic\\为什么老师会在这里\\12_work\\111\\汉化',
      imgMaxWidth: 200,
      imgMaxHeight: 280,
      similar_threshold: 0.5,
      maskUseCPU: true,
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
    },
    configTemp: {},
    draggedItem: null,
    btnDisable: false,
  }),
  methods: {
    resolveImgUrl(name, path) {
      return `/images/${this.config.imgMaxWidth}x${this.config.imgMaxHeight}/${path}\\${name}`
    },
    myRequest(url, data) {
      return fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json;charset=utf-8'
        },
        body: JSON.stringify(data)
      }).then(res => {
        return res.json()
      }).then(res => {
        if (res.code !== 200) {
          throw res['msg']
        } else {
          return res['data']
        }
      }).catch(err => {
        alert(err)
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
        this.$set(this.match_groups[idx], 'match', this.draggedItem)
        this.$set(this.match_groups[idx], 'matchRatio', 1)
        this.draggedItem = null
      }
    },
    clearMatch(idx) {
      this.$set(this.match_groups[idx], 'match', '')
      this.$set(this.match_groups[idx], 'matchRatio', 1)
    },
    configDialogOpen() {
      this.configTemp = {...this.config}
    },
    configDialogOk() {
      this.config = {...this.configTemp}
      this.configDialogShow = false
      localStorage.setItem('auto-ps-config', JSON.stringify(this.config))
    },
    startMatch() {
      this.btnDisable = true
      this.myRequest('/api/img_match', this.config).then(res => {
        this.match_groups = res['match_result']
        this.match_to_dirNum = res['match_from_num']
        this.match_from_dirNum = res['match_dir_num']
        this.match_from_list = res['match_from_list']
        localStorage.setItem('auto-ps-config', JSON.stringify(this.config))
      }).finally(_ => {
        this.btnDisable = false
      })
      this.match_to_dir = this.config['match_to_dir']
      this.match_from_dir = this.config['match_from_dir']
    },
    startPS() {
      if (this.match_to_dir !== this.config.match_to_dir || this.match_from_dir !== this.config.match_from_dir) {
        window.ELEMENT.Message({
          message: '当前显示的匹配结果和填入的路径不一致，无法继续！',
          type: 'warning',
          showClose: true,
          duration: 4000
        })
        return
      }
      this.btnDisable = true
      this.myRequest('/api/start_ps', {
        match_list: this.match_groups_result,
        config: this.config,
        task_id: `${this.config.match_to_dir}:${this.config.match_from_dir}`
      }).then(_ => {
        window.ELEMENT.Message({
          message: '任务下发成功，本页面不会提示任务进度！',
          type: 'success',
          showClose: true,
          duration: 4000
        })
      }).finally(_ => {
        this.btnDisable = false
      })
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